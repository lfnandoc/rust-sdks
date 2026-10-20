// Standalone probe for the Media Foundation H264 encoder spike (Livenalt
// ticket #199). No libwebrtc: it talks to the hardware encoder MFT directly with
// the same configuration the webrtc-sys MF encoder uses, and answers
//
//   1. which hardware H264 encoder MFTs exist and whether they are async,
//   2. whether the MFT accepts 3 temporal layers (CODECAPI_AVEncVideoTemporalLayerCount)
//      and how it marks the temporal id of each output (prefix NAL, sample
//      attribute, or not at all),
//   3. how many encode sessions can run concurrently (interleaved, like
//      SimulcastEncoderAdapter driving one MFT per simulcast layer).
//
// Build (x64 Developer Command Prompt):
//   cl /nologo /EHsc /std:c++20 /O2 mf_probe.cpp mfplat.lib mfuuid.lib ole32.lib oleaut32.lib
// Run:
//   mf_probe.exe [--sessions N] [--layers T] [--frames F] [--size WxH]
//                [--bitrate BPS] [--out DIR] [--scale-sessions]
//
// --scale-sessions creates sessions one at a time at 1920x1080 until one fails
// to activate, configure or encode (capped at 16) and prints the count.
// Each session's bitstream is written to DIR/session<i>_<W>x<H>.h264 (Annex B).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>

#include <strmif.h>

#include <initguid.h>

#include <codecapi.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

struct Options {
  int sessions = 3;
  int layers = 3;
  int frames = 90;
  int width = 1920;
  int height = 1080;
  uint32_t bitrate = 2'500'000;
  std::string out = ".";
  bool scale_sessions = false;
};

std::string Hr(HRESULT hr) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(hr));
  return buf;
}

std::string Narrow(const wchar_t* s, UINT32 len) {
  std::string out;
  for (UINT32 i = 0; i < len; i++) out.push_back(s[i] < 128 ? char(s[i]) : '?');
  return out;
}

std::string GuidString(const GUID& g) {
  wchar_t buf[64];
  StringFromGUID2(g, buf, 64);
  return Narrow(buf, static_cast<UINT32>(wcslen(buf)));
}

// Screen-like NV12 test image: white background, rows of 1-px "glyph" strokes
// that scroll by one text line every 15 frames, a blinking caret, and a small
// moving gradient box standing in for video.
void FillFrame(int n, int w, int h, int pitch, uint8_t* y, uint8_t* uv) {
  const int line_h = 18;
  const int scroll = (n / 15) * line_h;
  for (int r = 0; r < h; r++) {
    uint8_t* row = y + static_cast<size_t>(r) * pitch;
    const int text_row = (r + scroll) / line_h;
    const int in_line = (r + scroll) % line_h;
    for (int c = 0; c < w; c++) {
      uint8_t v = 235;
      if (in_line >= 4 && in_line < 14 && c >= 40 && c < w - 200) {
        const int glyph = (c - 40) / 9;
        const int gx = (c - 40) % 9;
        const uint32_t seed = static_cast<uint32_t>(text_row * 131 + glyph * 17);
        const bool space = (seed % 7) == 0 || glyph > 40 + int(seed % 60);
        if (!space && gx < 7) {
          const uint32_t bits = (seed * 2654435761u) >> (in_line + gx);
          if ((bits & 3) == 0 || gx == 0 || in_line == 13) v = 30;
        }
      }
      row[c] = v;
    }
  }
  // Caret.
  if ((n / 15) % 2 == 0) {
    for (int r = 200; r < 216 && r < h; r++) y[static_cast<size_t>(r) * pitch + 600] = 16;
  }
  // Moving gradient box.
  const int bx = 100 + (n * 7) % std::max(1, w - 400);
  const int by = h / 2;
  for (int r = by; r < by + 160 && r < h; r++)
    for (int c = bx; c < bx + 240 && c < w; c++)
      y[static_cast<size_t>(r) * pitch + c] = static_cast<uint8_t>((c - bx + r - by + n * 3) & 0xFF);
  for (int r = 0; r < h / 2; r++) {
    uint8_t* row = uv + static_cast<size_t>(r) * pitch;
    for (int c = 0; c < w; c += 2) {
      const bool in_box = r * 2 >= by && r * 2 < by + 160 && c >= bx && c < bx + 240;
      row[c] = in_box ? 90 : 128;
      row[c + 1] = in_box ? 200 : 128;
    }
  }
}

struct Nal {
  int type;
  size_t offset;  // of the NAL header byte
  size_t size;
};

std::vector<Nal> FindNals(const std::vector<uint8_t>& b) {
  std::vector<Nal> out;
  size_t i = 0;
  auto start_code_at = [&](size_t p, size_t* len) {
    if (p + 3 <= b.size() && b[p] == 0 && b[p + 1] == 0 && b[p + 2] == 1) {
      *len = 3;
      return true;
    }
    if (p + 4 <= b.size() && b[p] == 0 && b[p + 1] == 0 && b[p + 2] == 0 && b[p + 3] == 1) {
      *len = 4;
      return true;
    }
    return false;
  };
  std::vector<size_t> starts;
  while (i < b.size()) {
    size_t len = 0;
    if (start_code_at(i, &len)) {
      starts.push_back(i + len);
      i += len;
    } else {
      i++;
    }
  }
  for (size_t k = 0; k < starts.size(); k++) {
    size_t end = k + 1 < starts.size() ? starts[k + 1] : b.size();
    // Trim the next start code from this NAL's size.
    while (end > starts[k] && k + 1 < starts.size() && b[end - 1] == 0) end--;
    if (k + 1 < starts.size() && end >= 1 && b[end - 1] == 1) end--;
    while (end > starts[k] && k + 1 < starts.size() && b[end - 1] == 0) end--;
    if (starts[k] < b.size()) out.push_back({b[starts[k]] & 0x1F, starts[k], end - starts[k]});
  }
  return out;
}

// temporal_id from an H.264 SVC/MVC NAL header extension (NAL types 14 and 20):
// byte0 = NAL header, byte1 = svc_extension_flag(1) idr_flag(1) priority_id(6),
// byte2 = no_inter_layer_pred(1) dependency_id(3) quality_id(4),
// byte3 = temporal_id(3) use_ref_base(1) discardable(1) output(1) reserved(2).
int PrefixTemporalId(const std::vector<uint8_t>& b, const Nal& n) {
  if (n.size < 4) return -1;
  return (b[n.offset + 3] >> 5) & 0x7;
}

struct Session {
  int index = 0;
  ComPtr<IMFActivate> activate;
  ComPtr<IMFTransform> mft;
  ComPtr<ICodecAPI> api;
  ComPtr<IMFMediaEventGenerator> events;
  bool async = false;
  DWORD in_id = 0, out_id = 0;
  int credits = 0;
  int frames_in = 0, frames_out = 0;
  bool failed = false;
  std::string fail_reason;
  FILE* file = nullptr;
  std::map<int, int> nal_counts;
  std::map<int, int> tid_counts;        // from prefix NALs
  std::map<int, int> attr_tid_counts;   // from MFSampleExtension_TemporalLayerId
  std::vector<int> tid_sequence;        // first 16 outputs
  bool printed_attrs = false;
  double encode_ms = 0;
  size_t bytes = 0;
};

HRESULT SetU32(ICodecAPI* api, const GUID& g, UINT32 v) {
  VARIANT var;
  VariantInit(&var);
  var.vt = VT_UI4;
  var.ulVal = v;
  return api->SetValue(&g, &var);
}

HRESULT SetBool(ICodecAPI* api, const GUID& g, bool v) {
  VARIANT var;
  VariantInit(&var);
  var.vt = VT_BOOL;
  var.boolVal = v ? VARIANT_TRUE : VARIANT_FALSE;
  return api->SetValue(&g, &var);
}

std::vector<ComPtr<IMFActivate>> EnumHardwareEncoders() {
  MFT_REGISTER_TYPE_INFO in = {MFMediaType_Video, MFVideoFormat_NV12};
  MFT_REGISTER_TYPE_INFO out = {MFMediaType_Video, MFVideoFormat_H264};
  IMFActivate** acts = nullptr;
  UINT32 count = 0;
  HRESULT hr = MFTEnumEx(MFT_CATEGORY_VIDEO_ENCODER,
                         MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER, &in, &out,
                         &acts, &count);
  std::vector<ComPtr<IMFActivate>> v;
  if (FAILED(hr)) {
    std::printf("MFTEnumEx failed: %s\n", Hr(hr).c_str());
    return v;
  }
  for (UINT32 i = 0; i < count; i++) {
    v.emplace_back(acts[i]);
    acts[i]->Release();
  }
  CoTaskMemFree(acts);
  return v;
}

void PrintCodecApiProperty(ICodecAPI* api, const GUID& g, const char* name) {
  const bool supported = api->IsSupported(&g) == S_OK;
  const bool modifiable = api->IsModifiable(&g) == S_OK;
  std::printf("  %-36s supported=%d modifiable=%d", name, supported, modifiable);
  VARIANT v;
  VariantInit(&v);
  if (SUCCEEDED(api->GetValue(&g, &v))) {
    if (v.vt == VT_UI4) std::printf(" value=%lu", v.ulVal);
    else if (v.vt == VT_BOOL) std::printf(" value=%d", v.boolVal ? 1 : 0);
    else std::printf(" value(vt=%d)", v.vt);
  }
  VariantClear(&v);
  VARIANT lo, hi, step;
  VariantInit(&lo);
  VariantInit(&hi);
  VariantInit(&step);
  if (SUCCEEDED(api->GetParameterRange(&g, &lo, &hi, &step)) && lo.vt == VT_UI4)
    std::printf(" range=[%lu..%lu step %lu]", lo.ulVal, hi.ulVal, step.ulVal);
  std::printf("\n");
}

bool Fail(Session& s, const std::string& why, HRESULT hr) {
  s.failed = true;
  s.fail_reason = why + " " + Hr(hr);
  return false;
}

bool Configure(Session& s, const Options& o, int width, int height, int layers,
               const ComPtr<IMFActivate>& activate) {
  s.activate = activate;
  HRESULT hr = s.activate->ActivateObject(IID_PPV_ARGS(&s.mft));
  if (FAILED(hr)) return Fail(s, "ActivateObject", hr);
  ComPtr<IMFAttributes> attrs;
  if (SUCCEEDED(s.mft->GetAttributes(&attrs)) && attrs) {
    UINT32 a = 0;
    attrs->GetUINT32(MF_TRANSFORM_ASYNC, &a);
    s.async = a != 0;
    if (s.async) attrs->SetUINT32(MF_TRANSFORM_ASYNC_UNLOCK, TRUE);
  }
  if (s.async && FAILED(hr = s.mft.As(&s.events))) return Fail(s, "no event generator", hr);
  if (FAILED(hr = s.mft.As(&s.api))) return Fail(s, "no ICodecAPI", hr);
  DWORD in_ids[1] = {0}, out_ids[1] = {0};
  if (s.mft->GetStreamIDs(1, in_ids, 1, out_ids) == E_NOTIMPL) in_ids[0] = out_ids[0] = 0;
  s.in_id = in_ids[0];
  s.out_id = out_ids[0];

  // Same order as MFH264EncoderImpl::ConfigureTransform, with the temporal
  // layer count set first because several MFTs only accept it before the media
  // types are fixed.
  if (layers > 1) {
    hr = SetU32(s.api.Get(), CODECAPI_AVEncVideoTemporalLayerCount, layers);
    std::printf("  session %d: TemporalLayerCount=%d before types -> %s\n", s.index, layers,
                Hr(hr).c_str());
  }

  ComPtr<IMFMediaType> ot;
  MFCreateMediaType(&ot);
  ot->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
  ot->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
  ot->SetUINT32(MF_MT_AVG_BITRATE, o.bitrate);
  MFSetAttributeSize(ot.Get(), MF_MT_FRAME_SIZE, width, height);
  MFSetAttributeRatio(ot.Get(), MF_MT_FRAME_RATE, 30, 1);
  ot->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
  ot->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Base);
  ot->SetUINT32(MF_MT_MPEG2_LEVEL, eAVEncH264VLevel4_2);
  if (FAILED(hr = s.mft->SetOutputType(s.out_id, ot.Get(), 0))) {
    ot->DeleteItem(MF_MT_MPEG2_LEVEL);
    if (FAILED(hr = s.mft->SetOutputType(s.out_id, ot.Get(), 0)))
      return Fail(s, "SetOutputType", hr);
  }
  ComPtr<IMFMediaType> it;
  for (DWORD i = 0;; i++) {
    ComPtr<IMFMediaType> c;
    if (FAILED(s.mft->GetInputAvailableType(s.in_id, i, &c))) break;
    GUID sub = {};
    c->GetGUID(MF_MT_SUBTYPE, &sub);
    if (sub == MFVideoFormat_NV12) {
      it = c;
      break;
    }
  }
  if (!it) {
    MFCreateMediaType(&it);
    it->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    it->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
  }
  MFSetAttributeSize(it.Get(), MF_MT_FRAME_SIZE, width, height);
  MFSetAttributeRatio(it.Get(), MF_MT_FRAME_RATE, 30, 1);
  it->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
  if (FAILED(hr = s.mft->SetInputType(s.in_id, it.Get(), 0))) return Fail(s, "SetInputType", hr);

  if (FAILED(hr = SetU32(s.api.Get(), CODECAPI_AVEncCommonRateControlMode,
                         eAVEncCommonRateControlMode_CBR)))
    return Fail(s, "CBR", hr);
  if (FAILED(hr = SetU32(s.api.Get(), CODECAPI_AVEncCommonMeanBitRate, o.bitrate)))
    return Fail(s, "MeanBitRate", hr);
  // PR #1282 treats a rejected BPictureCount=0 as fatal. Report what the MFT
  // says and only fail if B-frames are actually on.
  if (FAILED(hr = SetU32(s.api.Get(), CODECAPI_AVEncMPVDefaultBPictureCount, 0))) {
    VARIANT v;
    VariantInit(&v);
    const HRESULT ghr = s.api->GetValue(&CODECAPI_AVEncMPVDefaultBPictureCount, &v);
    std::printf("  session %d: BPictureCount=0 rejected %s; read back %s value=%lu\n", s.index,
                Hr(hr).c_str(), Hr(ghr).c_str(), v.vt == VT_UI4 ? v.ulVal : 0xFFFFFFFF);
    if (SUCCEEDED(ghr) && v.vt == VT_UI4 && v.ulVal != 0) return Fail(s, "B-frames on", hr);
  }
  SetBool(s.api.Get(), CODECAPI_AVLowLatencyMode, true);
  SetU32(s.api.Get(), CODECAPI_AVEncMPVGOPSize, 0x7FFFFFFF);
  if (layers > 1) {
    VARIANT v;
    VariantInit(&v);
    hr = s.api->GetValue(&CODECAPI_AVEncVideoTemporalLayerCount, &v);
    std::printf("  session %d: TemporalLayerCount read back after types -> %s value=%lu\n",
                s.index, Hr(hr).c_str(), v.vt == VT_UI4 ? v.ulVal : 0);
    if (FAILED(hr) || v.vt != VT_UI4 || v.ulVal != static_cast<ULONG>(layers)) {
      hr = SetU32(s.api.Get(), CODECAPI_AVEncVideoTemporalLayerCount, layers);
      std::printf("  session %d: TemporalLayerCount=%d after types -> %s\n", s.index, layers,
                  Hr(hr).c_str());
    }
  }
  if (FAILED(hr = s.mft->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)))
    return Fail(s, "BEGIN_STREAMING", hr);
  if (FAILED(hr = s.mft->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)))
    return Fail(s, "START_OF_STREAM", hr);
  return true;
}

void DumpSampleAttributes(Session& s, IMFSample* sample) {
  UINT32 n = 0;
  sample->GetCount(&n);
  std::printf("  session %d: output sample carries %u attribute(s)\n", s.index, n);
  for (UINT32 i = 0; i < n; i++) {
    GUID key;
    PROPVARIANT v;
    PropVariantInit(&v);
    if (SUCCEEDED(sample->GetItemByIndex(i, &key, &v))) {
      std::printf("    %s vt=%d", GuidString(key).c_str(), v.vt);
      if (v.vt == VT_UI4) std::printf(" u32=%lu", v.ulVal);
      if (v.vt == VT_UI8) std::printf(" u64=%llu", v.uhVal.QuadPart);
      std::printf("\n");
    }
    PropVariantClear(&v);
  }
}

bool CollectOne(Session& s, int width, int height) {
  MFT_OUTPUT_STREAM_INFO info = {};
  HRESULT hr = s.mft->GetOutputStreamInfo(s.out_id, &info);
  if (FAILED(hr)) return Fail(s, "GetOutputStreamInfo", hr);
  const bool provides =
      info.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES);
  ComPtr<IMFSample> allocated;
  if (!provides) {
    MFCreateSample(&allocated);
    ComPtr<IMFMediaBuffer> buf;
    MFCreateMemoryBuffer(info.cbSize ? info.cbSize : width * height * 3 / 2, &buf);
    allocated->AddBuffer(buf.Get());
  }
  MFT_OUTPUT_DATA_BUFFER out = {};
  out.dwStreamID = s.out_id;
  out.pSample = provides ? nullptr : allocated.Get();
  DWORD status = 0;
  hr = s.mft->ProcessOutput(0, 1, &out, &status);
  if (out.pEvents) out.pEvents->Release();
  if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return true;
  if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
    for (DWORD i = 0;; i++) {
      ComPtr<IMFMediaType> c;
      if (FAILED(s.mft->GetOutputAvailableType(s.out_id, i, &c))) break;
      GUID sub = {};
      c->GetGUID(MF_MT_SUBTYPE, &sub);
      if (sub == MFVideoFormat_H264) {
        s.mft->SetOutputType(s.out_id, c.Get(), 0);
        break;
      }
    }
    return true;
  }
  if (FAILED(hr)) return Fail(s, "ProcessOutput", hr);
  ComPtr<IMFSample> sample;
  if (provides) sample.Attach(out.pSample);
  else sample = allocated;
  if (!sample) return true;
  if (!s.printed_attrs) {
    DumpSampleAttributes(s, sample.Get());
    s.printed_attrs = true;
  }
  UINT32 attr_tid = 0;
  if (SUCCEEDED(sample->GetUINT32(MFSampleExtension_TemporalLayerId, &attr_tid)))
    s.attr_tid_counts[static_cast<int>(attr_tid)]++;
  ComPtr<IMFMediaBuffer> cb;
  sample->ConvertToContiguousBuffer(&cb);
  BYTE* data = nullptr;
  DWORD max_len = 0, len = 0;
  cb->Lock(&data, &max_len, &len);
  std::vector<uint8_t> bytes(data, data + len);
  cb->Unlock();
  s.frames_out++;
  s.bytes += bytes.size();
  if (s.file) fwrite(bytes.data(), 1, bytes.size(), s.file);
  int tid = -1;
  for (const Nal& n : FindNals(bytes)) {
    s.nal_counts[n.type]++;
    if (n.type == 14 || n.type == 20) tid = PrefixTemporalId(bytes, n);
  }
  if (tid >= 0) s.tid_counts[tid]++;
  if (s.tid_sequence.size() < 16) s.tid_sequence.push_back(tid);
  return true;
}

// Pumps one session until it has an input credit (async) or returns.
bool PumpUntilInput(Session& s, int width, int height, int timeout_ms) {
  if (!s.async) return true;
  const ULONGLONG deadline = GetTickCount64() + timeout_ms;
  while (s.credits == 0) {
    ComPtr<IMFMediaEvent> ev;
    HRESULT hr = s.events->GetEvent(MF_EVENT_FLAG_NO_WAIT, &ev);
    if (hr == MF_E_NO_EVENTS_AVAILABLE) {
      if (GetTickCount64() > deadline) return Fail(s, "timeout waiting for input", hr);
      Sleep(1);
      continue;
    }
    if (FAILED(hr)) return Fail(s, "GetEvent", hr);
    MediaEventType t = MEUnknown;
    ev->GetType(&t);
    if (t == METransformNeedInput) s.credits++;
    else if (t == METransformHaveOutput && !CollectOne(s, width, height)) return false;
  }
  return true;
}

void DrainAvailable(Session& s, int width, int height) {
  if (!s.async) {
    while (!s.failed) {
      const int before = s.frames_out;
      CollectOne(s, width, height);
      if (s.frames_out == before) break;
    }
    return;
  }
  for (;;) {
    ComPtr<IMFMediaEvent> ev;
    if (s.events->GetEvent(MF_EVENT_FLAG_NO_WAIT, &ev) != S_OK) return;
    MediaEventType t = MEUnknown;
    ev->GetType(&t);
    if (t == METransformNeedInput) s.credits++;
    else if (t == METransformHaveOutput && !CollectOne(s, width, height)) return;
  }
}

bool EncodeFrame(Session& s, int n, int width, int height, bool force_key) {
  if (!PumpUntilInput(s, width, height, 1000)) return false;
  ComPtr<IMFMediaBuffer> mb;
  HRESULT hr = MFCreate2DMediaBuffer(width, height, MFVideoFormat_NV12.Data1, FALSE, &mb);
  if (FAILED(hr)) return Fail(s, "MFCreate2DMediaBuffer", hr);
  ComPtr<IMF2DBuffer> b2;
  mb.As(&b2);
  BYTE* scan0 = nullptr;
  LONG pitch = 0;
  b2->Lock2D(&scan0, &pitch);
  FillFrame(n, width, height, pitch, scan0, scan0 + static_cast<size_t>(pitch) * height);
  b2->Unlock2D();
  DWORD cl = 0;
  if (SUCCEEDED(b2->GetContiguousLength(&cl))) mb->SetCurrentLength(cl);
  ComPtr<IMFSample> sample;
  MFCreateSample(&sample);
  sample->AddBuffer(mb.Get());
  sample->SetSampleTime(static_cast<LONGLONG>(n) * 333333);
  sample->SetSampleDuration(333333);
  if (force_key) SetU32(s.api.Get(), CODECAPI_AVEncVideoForceKeyFrame, 1);
  const auto t0 = std::chrono::steady_clock::now();
  hr = s.mft->ProcessInput(s.in_id, sample.Get(), 0);
  if (hr == MF_E_NOTACCEPTING && !s.async) {
    DrainAvailable(s, width, height);
    hr = s.mft->ProcessInput(s.in_id, sample.Get(), 0);
  }
  if (FAILED(hr)) return Fail(s, "ProcessInput", hr);
  if (s.async) s.credits--;
  s.frames_in++;
  DrainAvailable(s, width, height);
  s.encode_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return true;
}

void Finish(Session& s, int width, int height) {
  if (!s.mft) return;
  s.mft->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, s.in_id);
  s.mft->ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0);
  const ULONGLONG deadline = GetTickCount64() + 1000;
  while (s.frames_out < s.frames_in && GetTickCount64() < deadline && !s.failed) {
    DrainAvailable(s, width, height);
    Sleep(1);
  }
  s.mft->ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
  s.events.Reset();
  s.api.Reset();
  s.mft.Reset();
  if (s.activate) s.activate->ShutdownObject();
  if (s.file) fclose(s.file);
  s.file = nullptr;
}

void PrintSession(const Session& s, int width, int height) {
  std::printf("session %d %dx%d: in=%d out=%d bytes=%zu avg_submit_ms=%.2f %s%s\n", s.index,
              width, height, s.frames_in, s.frames_out, s.bytes,
              s.frames_in ? s.encode_ms / s.frames_in : 0.0, s.failed ? "FAILED: " : "ok",
              s.fail_reason.c_str());
  std::printf("  NAL types:");
  for (auto& [t, c] : s.nal_counts) std::printf(" %d:%d", t, c);
  std::printf("\n  temporal ids from prefix NALs:");
  if (s.tid_counts.empty()) std::printf(" none");
  for (auto& [t, c] : s.tid_counts) std::printf(" T%d:%d", t, c);
  std::printf("\n  temporal ids from MFSampleExtension_TemporalLayerId:");
  if (s.attr_tid_counts.empty()) std::printf(" none");
  for (auto& [t, c] : s.attr_tid_counts) std::printf(" T%d:%d", t, c);
  std::printf("\n  first temporal ids:");
  for (int t : s.tid_sequence) std::printf(" %d", t);
  std::printf("\n");
}

int Run(const Options& o) {
  auto encoders = EnumHardwareEncoders();
  std::printf("hardware H264 encoder MFTs: %zu\n", encoders.size());
  for (auto& a : encoders) {
    WCHAR* name = nullptr;
    UINT32 len = 0;
    std::string n = "?";
    if (SUCCEEDED(a->GetAllocatedString(MFT_FRIENDLY_NAME_Attribute, &name, &len))) {
      n = Narrow(name, len);
      CoTaskMemFree(name);
    }
    WCHAR* vendor = nullptr;
    std::string v = "?";
    if (SUCCEEDED(a->GetAllocatedString(MFT_ENUM_HARDWARE_VENDOR_ID_Attribute, &vendor, &len))) {
      v = Narrow(vendor, len);
      CoTaskMemFree(vendor);
    }
    std::printf("  %s (vendor %s)\n", n.c_str(), v.c_str());
  }
  if (encoders.empty()) return 2;

  {
    // Capability report on a throwaway activation of the first MFT.
    ComPtr<IMFTransform> mft;
    if (SUCCEEDED(encoders[0]->ActivateObject(IID_PPV_ARGS(&mft)))) {
      ComPtr<ICodecAPI> api;
      if (SUCCEEDED(mft.As(&api))) {
        std::printf("ICodecAPI on first MFT:\n");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncVideoTemporalLayerCount,
                              "AVEncVideoTemporalLayerCount");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncVideoMaxTemporalLayers,
                              "AVEncVideoMaxTemporalLayers");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncVideoSelectLayer, "AVEncVideoSelectLayer");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncVideoLTRBufferControl,
                              "AVEncVideoLTRBufferControl");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncVideoMaxNumRefFrame,
                              "AVEncVideoMaxNumRefFrame");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncCommonQualityVsSpeed,
                              "AVEncCommonQualityVsSpeed");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVEncVideoContentType, "AVEncVideoContentType");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVScenarioInfo, "AVScenarioInfo");
        PrintCodecApiProperty(api.Get(), CODECAPI_AVLowLatencyMode, "AVLowLatencyMode");
      }
    }
    encoders[0]->ShutdownObject();
    encoders = EnumHardwareEncoders();
  }

  if (o.scale_sessions) {
    std::vector<std::unique_ptr<Session>> sessions;
    for (int i = 0; i < 16; i++) {
      auto acts = EnumHardwareEncoders();
      auto s = std::make_unique<Session>();
      s->index = i;
      if (!Configure(*s, o, o.width, o.height, o.layers, acts[0])) {
        std::printf("session %d could not be configured: %s\n", i, s->fail_reason.c_str());
        break;
      }
      sessions.push_back(std::move(s));
      bool all_ok = true;
      const int base = static_cast<int>(sessions.size()) * 1000;
      for (int f = 0; f < 10 && all_ok; f++)
        for (auto& x : sessions)
          if (!EncodeFrame(*x, base + f, o.width, o.height, f == 0)) all_ok = false;
      if (!all_ok) {
        for (auto& x : sessions)
          if (x->failed)
            std::printf("with %zu sessions, session %d failed: %s\n", sessions.size(), x->index,
                        x->fail_reason.c_str());
        break;
      }
      std::printf("%zu concurrent sessions at %dx%d: all encoding\n", sessions.size(), o.width,
                  o.height);
    }
    for (auto& s : sessions) Finish(*s, o.width, o.height);
    return 0;
  }

  // Simulcast-like run: session i encodes at size / 2^(sessions-1-i), all
  // sessions interleaved frame by frame from the same source frames.
  std::vector<std::unique_ptr<Session>> sessions;
  std::vector<std::pair<int, int>> sizes;
  for (int i = 0; i < o.sessions; i++) {
    const int div = 1 << (o.sessions - 1 - i);
    const int w = (o.width / div) & ~1;
    const int h = (o.height / div) & ~1;
    auto acts = EnumHardwareEncoders();
    auto s = std::make_unique<Session>();
    s->index = i;
    const std::string path =
        o.out + "\\session" + std::to_string(i) + "_" + std::to_string(w) + "x" +
        std::to_string(h) + "_L1T" + std::to_string(o.layers) + ".h264";
    s->file = fopen(path.c_str(), "wb");
    if (!Configure(*s, o, w, h, o.layers, acts[0]))
      std::printf("session %d configure failed: %s\n", i, s->fail_reason.c_str());
    sessions.push_back(std::move(s));
    sizes.push_back({w, h});
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int f = 0; f < o.frames; f++)
    for (size_t i = 0; i < sessions.size(); i++)
      if (!sessions[i]->failed)
        EncodeFrame(*sessions[i], f, sizes[i].first, sizes[i].second, f == 0);
  const double wall =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  for (size_t i = 0; i < sessions.size(); i++) Finish(*sessions[i], sizes[i].first, sizes[i].second);
  for (size_t i = 0; i < sessions.size(); i++) PrintSession(*sessions[i], sizes[i].first, sizes[i].second);
  std::printf("wall time for %d frames x %zu sessions: %.0f ms (%.2f ms per frame set)\n", o.frames,
              sessions.size(), wall, wall / o.frames);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--sessions") o.sessions = std::stoi(next());
    else if (a == "--layers") o.layers = std::stoi(next());
    else if (a == "--frames") o.frames = std::stoi(next());
    else if (a == "--bitrate") o.bitrate = static_cast<uint32_t>(std::stoul(next()));
    else if (a == "--out") o.out = next();
    else if (a == "--scale-sessions") o.scale_sessions = true;
    else if (a == "--size") {
      std::string s = next();
      std::sscanf(s.c_str(), "%dx%d", &o.width, &o.height);
    }
  }
  if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return 1;
  if (FAILED(MFStartup(MF_VERSION, MFSTARTUP_LITE))) return 1;
  const int rc = Run(o);
  MFShutdown();
  CoUninitialize();
  return rc;
}
