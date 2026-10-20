// Spike harness for the Media Foundation H264 encoder (Livenalt ticket #199).
//
// The pure helpers at the top run anywhere. The tests marked #[ignore] need the
// native libwebrtc and a GPU with a hardware H264 encoder MFT; run them by hand.

#![cfg(not(target_arch = "wasm32"))]

/// Frame index barcode drawn into the top-left of every source frame so a
/// decoded frame can be paired with the source frame it came from.
mod barcode {
    /// 8 marker bits then 24 index bits, one CELL x CELL square per bit.
    pub const BITS: usize = 32;
    pub const CELL: usize = 24;
    const MARKER: u32 = 0xA5;

    pub fn write(y: &mut [u8], stride: usize, index: u32) {
        let word = (MARKER << 24) | (index & 0x00FF_FFFF);
        for bit in 0..BITS {
            let v = if word >> (BITS - 1 - bit) & 1 == 1 { 16 } else { 235 };
            for r in 0..CELL {
                let row = r * stride + bit * CELL;
                y[row..row + CELL].fill(v);
            }
        }
    }

    pub fn read(y: &[u8], stride: usize) -> Option<u32> {
        let mut word = 0u32;
        for bit in 0..BITS {
            // Average the middle half of the cell; edges blur under compression.
            let mut sum = 0u32;
            for r in CELL / 4..CELL * 3 / 4 {
                let row = r * stride + bit * CELL;
                sum += y[row + CELL / 4..row + CELL * 3 / 4].iter().map(|&p| p as u32).sum::<u32>();
            }
            let mean = sum / ((CELL / 2) * (CELL / 2)) as u32;
            word = (word << 1) | u32::from(mean < 128);
        }
        (word >> 24 == MARKER).then_some(word & 0x00FF_FFFF)
    }
}

/// PSNR in dB of two 8-bit planes of the same size; identical planes give
/// `f64::INFINITY`.
fn psnr(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> f64 {
    let mut sse = 0u64;
    for r in 0..h {
        let ra = &a[r * a_stride..r * a_stride + w];
        let rb = &b[r * b_stride..r * b_stride + w];
        sse += ra.iter().zip(rb).map(|(&x, &y)| (x as i64 - y as i64).pow(2) as u64).sum::<u64>();
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / (w * h) as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Per-session temporal-layer counts the MF encoder logs every 300 frames and
/// on release, as
/// `MF H264 temporal ids 1920x1080 layers=3 frames=120 t0=30 t1=30 t2=60 untagged=0`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TemporalSummary {
    width: u32,
    height: u32,
    layers: u32,
    frames: u64,
    per_layer: [u64; 3],
    untagged: u64,
}

fn parse_temporal_summary(line: &str) -> Option<TemporalSummary> {
    let rest = line.split_once("MF H264 temporal ids ")?.1;
    let mut words = rest.split_whitespace();
    let (w, h) = words.next()?.split_once('x')?;
    let mut fields = std::collections::HashMap::new();
    for word in words {
        let (k, v) = word.split_once('=')?;
        fields.insert(k, v.parse::<u64>().ok()?);
    }
    let get = |k: &str| fields.get(k).copied();
    Some(TemporalSummary {
        width: w.parse().ok()?,
        height: h.parse().ok()?,
        layers: get("layers")? as u32,
        frames: get("frames")?,
        per_layer: [get("t0")?, get("t1")?, get("t2")?],
        untagged: get("untagged")?,
    })
}

#[test]
fn temporal_summary_is_parsed_from_a_webrtc_log_line() {
    let line = "(h264_encoder_impl.cpp:1012): MF H264 temporal ids 960x540 layers=3 \
                frames=120 t0=30 t1=30 t2=60 untagged=0";
    assert_eq!(
        parse_temporal_summary(line),
        Some(TemporalSummary {
            width: 960,
            height: 540,
            layers: 3,
            frames: 120,
            per_layer: [30, 30, 60],
            untagged: 0,
        })
    );
    assert_eq!(parse_temporal_summary("MediaFoundation H264 encoder initialized"), None);
    assert_eq!(parse_temporal_summary("MF H264 temporal ids garbage"), None);
}

#[test]
fn psnr_matches_worked_examples() {
    let (w, h) = (16usize, 8usize);
    let a = vec![100u8; w * h];
    assert_eq!(psnr(&a, w, &a, w, w, h), f64::INFINITY);
    // MSE 1: 10 * log10(255^2) = 48.1308 dB.
    let b = vec![101u8; w * h];
    assert!((psnr(&a, w, &b, w, w, h) - 48.1308).abs() < 1e-3);
    // Full-scale error everywhere: 0 dB.
    assert!(psnr(&vec![0u8; w * h], w, &vec![255u8; w * h], w, w, h).abs() < 1e-9);
    // Stride padding is ignored: only the first w bytes of each row count.
    let mut padded = vec![7u8; (w + 4) * h];
    for r in 0..h {
        padded[r * (w + 4)..r * (w + 4) + w].fill(100);
    }
    assert_eq!(psnr(&a, w, &padded, w + 4, w, h), f64::INFINITY);
}

#[test]
fn barcode_round_trips_a_frame_index() {
    let (w, h) = (1920usize, 64usize);
    for index in [0u32, 1, 2, 1234, 0x00FF_FFFF] {
        let mut y = vec![235u8; w * h];
        barcode::write(&mut y, w, index);
        assert_eq!(barcode::read(&y, w), Some(index));
    }
}

#[test]
fn barcode_is_absent_on_a_plain_frame() {
    let (w, h) = (1920usize, 64usize);
    assert_eq!(barcode::read(&vec![235u8; w * h], w), None);
    assert_eq!(barcode::read(&vec![16u8; w * h], w), None);
}

// ---------------------------------------------------------------------------
// Synthetic screen content. Deterministic in the frame index, so a decoded
// frame can be compared with the exact source frame it came from.
// ---------------------------------------------------------------------------

/// A tall "document" of 1-px glyph strokes on white, some lines coloured, that
/// the source window scrolls through one text line every 15 frames, plus a
/// 240x160 moving gradient box standing in for video, a blinking caret and the
/// frame-index barcode.
struct ScreenSource {
    width: usize,
    height: usize,
    page_y: Vec<u8>,
    page_u: Vec<u8>,
    page_v: Vec<u8>,
}

const LINE_H: usize = 18;

impl ScreenSource {
    fn new(width: usize, height: usize) -> Self {
        let page_h = height * 3;
        let mut page_y = vec![235u8; width * page_h];
        let (cw, ch) = (width / 2, page_h / 2);
        let mut page_u = vec![128u8; cw * ch];
        let mut page_v = vec![128u8; cw * ch];
        for r in 0..page_h {
            let line = r / LINE_H;
            let in_line = r % LINE_H;
            if !(4..14).contains(&in_line) {
                continue;
            }
            let indent = 40 + (line % 5) * 36;
            let len = 30 + (line * 37 % 150);
            let coloured = line % 4 == 1;
            for glyph in 0..len {
                let x0 = indent + glyph * 9;
                if x0 + 9 >= width - 40 {
                    break;
                }
                let seed = (line as u32).wrapping_mul(131).wrapping_add(glyph as u32 * 17);
                if seed % 7 == 0 {
                    continue; // space
                }
                for gx in 0..7usize {
                    let bits = seed.wrapping_mul(2_654_435_761) >> ((in_line + gx) % 29);
                    if bits & 3 == 0 || gx == 0 || in_line == 13 {
                        page_y[r * width + x0 + gx] = if coloured { 81 } else { 30 };
                        if coloured {
                            page_u[(r / 2) * cw + (x0 + gx) / 2] = 90;
                            page_v[(r / 2) * cw + (x0 + gx) / 2] = 240;
                        }
                    }
                }
            }
        }
        Self { width, height, page_y, page_u, page_v }
    }

    /// Writes frame `n` into the given I420 planes.
    #[allow(clippy::too_many_arguments)]
    fn render(
        &self,
        n: u32,
        y: &mut [u8],
        sy: usize,
        u: &mut [u8],
        su: usize,
        v: &mut [u8],
        sv: usize,
    ) {
        let (w, h) = (self.width, self.height);
        let page_rows = self.page_y.len() / w;
        // Scroll by whole line pairs so the chroma window stays aligned.
        let scroll = ((n as usize / 15) * LINE_H) % (page_rows - h);
        for r in 0..h {
            y[r * sy..r * sy + w]
                .copy_from_slice(&self.page_y[(r + scroll) * w..(r + scroll + 1) * w]);
        }
        let (cw, scroll_c) = (w / 2, scroll / 2);
        for r in 0..h / 2 {
            u[r * su..r * su + cw]
                .copy_from_slice(&self.page_u[(r + scroll_c) * cw..(r + scroll_c + 1) * cw]);
            v[r * sv..r * sv + cw]
                .copy_from_slice(&self.page_v[(r + scroll_c) * cw..(r + scroll_c + 1) * cw]);
        }
        if (n / 15) % 2 == 0 {
            for r in 200..216.min(h) {
                y[r * sy + 600.min(w - 1)] = 16;
            }
        }
        let bx = 100 + (n as usize * 7) % (w - 400);
        let by = h / 2;
        for r in by..(by + 160).min(h) {
            for c in bx..bx + 240 {
                y[r * sy + c] = ((c - bx + r - by + n as usize * 3) & 0xFF) as u8;
            }
        }
        for r in by / 2..((by + 160) / 2).min(h / 2) {
            u[r * su + bx / 2..r * su + (bx + 240) / 2].fill(90);
            v[r * sv + bx / 2..r * sv + (bx + 240) / 2].fill(200);
        }
        barcode::write(y, sy, n);
    }
}

#[test]
fn screen_source_frames_carry_their_index() {
    let src = ScreenSource::new(1280, 720);
    let (mut y, mut u, mut v) = (vec![0u8; 1280 * 720], vec![0u8; 640 * 360], vec![0u8; 640 * 360]);
    for n in [0u32, 14, 15, 299, 1000] {
        src.render(n, &mut y, 1280, &mut u, 640, &mut v, 640);
        assert_eq!(barcode::read(&y, 1280), Some(n));
    }
}

// ---------------------------------------------------------------------------
// Minimal PNG writer (stored deflate blocks) for the hand-review stills.
// ---------------------------------------------------------------------------

type Planes<'a> = (&'a [u8], usize, &'a [u8], usize, &'a [u8], usize);

fn write_png_i420(path: &std::path::Path, w: usize, h: usize, planes: Planes<'_>) {
    let (y, sy, u, su, v, sv) = planes;
    let mut raw = Vec::with_capacity((w * 3 + 1) * h);
    for r in 0..h {
        raw.push(0u8); // filter: none
        for c in 0..w {
            // BT.601 limited range, what libwebrtc's I420 frames use.
            let yy = (y[r * sy + c] as f32 - 16.0) * 1.164;
            let uu = u[(r / 2) * su + c / 2] as f32 - 128.0;
            let vv = v[(r / 2) * sv + c / 2] as f32 - 128.0;
            let px = [yy + 1.596 * vv, yy - 0.392 * uu - 0.813 * vv, yy + 2.017 * uu];
            raw.extend(px.iter().map(|p| p.round().clamp(0.0, 255.0) as u8));
        }
    }
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend((data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend(kind);
        out.extend(data);
        let crc = crc32(&out[start..]);
        out.extend(crc.to_be_bytes());
    }
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, block) in blocks.iter().enumerate() {
        z.push(u8::from(i + 1 == blocks.len()));
        let len = block.len() as u16;
        z.extend(len.to_le_bytes());
        z.extend((!len).to_le_bytes());
        z.extend(*block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend(((b << 16) | a).to_be_bytes());
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend((w as u32).to_be_bytes());
    ihdr.extend((h as u32).to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png).expect("write png");
}

// ---------------------------------------------------------------------------
// GPU tests: real libwebrtc, two peer connections in one process.
// ---------------------------------------------------------------------------

mod rig {
    use super::*;
    use libwebrtc::{
        peer_connection_factory::native::PeerConnectionFactoryExt,
        prelude::*,
        stats::{OutboundRtpStats, RtcStats},
        video_source::native::NativeVideoSource,
        video_stream::native::NativeVideoStream,
        video_track::ContentHint,
    };
    pub use std::sync::atomic::AtomicBool as Flag;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    };
    use std::time::{Duration, Instant};
    use tokio_stream::StreamExt;

    /// Keeps the libwebrtc log lines that matter for the encoder.
    struct Capture;
    static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
    const KEEP: &[&str] = &[
        "MediaFoundation",
        "MF H264",
        "encoder MFT",
        "Encoder MFT",
        "BPicture",
        "B-frames",
        "TemporalLayer",
        "temporal layer",
        "Failed to initialize",
        "falling back",
        "fallback",
        "Fallback",
        "SimulcastEncoderAdapter",
        "switch",
        "OpenH264",
        "InitEncode",
    ];

    impl log::Log for Capture {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, record: &log::Record) {
            let msg = record.args().to_string();
            if KEEP.iter().any(|k| msg.contains(k)) {
                LINES.lock().unwrap().push(msg);
            }
        }
        fn flush(&self) {}
    }

    pub fn init_logging() {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            log::set_boxed_logger(Box::new(Capture)).expect("logger");
            log::set_max_level(log::LevelFilter::Debug);
        });
    }

    pub fn take_lines() -> Vec<String> {
        std::mem::take(&mut *LINES.lock().unwrap())
    }

    pub fn print_lines(lines: &[String], max: usize) {
        let mut seen = std::collections::BTreeSet::new();
        for l in lines {
            let key: String = l.chars().take(160).collect();
            if seen.insert(key.clone()) && seen.len() <= max {
                println!("   log: {key}");
            }
        }
    }

    pub fn temporal_summaries(lines: &[String]) -> Vec<TemporalSummary> {
        lines.iter().filter_map(|l| parse_temporal_summary(l)).collect()
    }

    pub fn out_dir() -> std::path::PathBuf {
        let dir = std::env::var("MF_SPIKE_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("mf_h264_spike"));
        std::fs::create_dir_all(&dir).expect("create out dir");
        dir
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Codec {
        H264,
        Vp8,
    }

    pub struct Pair {
        pub sender: PeerConnection,
        pub receiver: PeerConnection,
        pub source: NativeVideoSource,
    }

    pub struct Layer {
        pub rid: &'static str,
        pub scale: f64,
        pub max_bitrate: u64,
    }

    /// The production 1080p ladder (packages/rtc/src/sfuLayers.ts in the
    /// Livenalt repo), L1T3 on every encoding as mediasoup-client sets it.
    pub const LADDER: [Layer; 3] = [
        Layer { rid: "q", scale: 4.0, max_bitrate: 300_000 },
        Layer { rid: "h", scale: 2.0, max_bitrate: 1_200_000 },
        Layer { rid: "f", scale: 1.0, max_bitrate: 4_000_000 },
    ];

    fn codec_preferences(factory: &PeerConnectionFactory, codec: Codec) -> Vec<RtpCodecCapability> {
        let caps = factory.get_rtp_sender_capabilities(MediaType::Video);
        let mut chosen: Vec<RtpCodecCapability> = caps
            .codecs
            .iter()
            .filter(|c| match codec {
                Codec::Vp8 => c.mime_type.eq_ignore_ascii_case("video/VP8"),
                Codec::H264 => {
                    let fmtp = c.sdp_fmtp_line.clone().unwrap_or_default();
                    c.mime_type.eq_ignore_ascii_case("video/H264")
                        && fmtp.contains("packetization-mode=1")
                        && fmtp.contains("profile-level-id=42e0")
                }
            })
            .cloned()
            .collect();
        assert!(!chosen.is_empty(), "no {codec:?} sender capability: {:?}", caps.codecs);
        chosen.extend(
            caps.codecs.iter().filter(|c| c.mime_type.eq_ignore_ascii_case("video/rtx")).cloned(),
        );
        chosen
    }

    pub async fn connect(
        factory: &PeerConnectionFactory,
        codec: Codec,
        layers: &[Layer],
        width: u32,
        height: u32,
        scalability: &str,
    ) -> Pair {
        let sender = factory.create_peer_connection(RtcConfiguration::default()).unwrap();
        let receiver = factory.create_peer_connection(RtcConfiguration::default()).unwrap();

        let source = NativeVideoSource::new(VideoResolution { width, height }, true);
        let track = factory.create_video_track("screen", source.clone());
        track.set_content_hint(ContentHint::Text);
        let encodings = layers
            .iter()
            .map(|l| RtpEncodingParameters {
                rid: if layers.len() > 1 { l.rid.to_string() } else { String::new() },
                scale_resolution_down_by: Some(l.scale),
                max_bitrate: Some(l.max_bitrate),
                scalability_mode: Some(scalability.to_string()),
                ..Default::default()
            })
            .collect();
        let transceiver = sender
            .add_transceiver(
                MediaStreamTrack::Video(track),
                RtpTransceiverInit {
                    direction: RtpTransceiverDirection::SendOnly,
                    stream_ids: vec!["screen".into()],
                    send_encodings: encodings,
                },
            )
            .unwrap();
        transceiver.set_codec_preferences(codec_preferences(factory, codec)).unwrap();

        let to_receiver: Arc<Mutex<Vec<IceCandidate>>> = Default::default();
        let to_sender: Arc<Mutex<Vec<IceCandidate>>> = Default::default();
        {
            let q = to_receiver.clone();
            sender.on_ice_candidate(Some(Box::new(move |c| q.lock().unwrap().push(c))));
            let q = to_sender.clone();
            receiver.on_ice_candidate(Some(Box::new(move |c| q.lock().unwrap().push(c))));
        }

        let offer = sender.create_offer(OfferOptions::default()).await.unwrap();
        let sender_offer_sdp = offer.to_string();
        sender.set_local_description(offer.clone()).await.unwrap();
        receiver.set_remote_description(offer).await.unwrap();
        let answer = receiver.create_answer(AnswerOptions::default()).await.unwrap();
        receiver.set_local_description(answer.clone()).await.unwrap();
        if std::env::var("MF_SPIKE_DEBUG_SDP").is_ok() {
            for (side, sdp) in
                [("offer", sender_offer_sdp.as_str()), ("answer", &answer.to_string())]
            {
                for l in sdp.lines().filter(|l| l.contains("simulcast") || l.contains("a=rid")) {
                    println!("   sdp {side}: {l}");
                }
            }
        }
        // libwebrtc's answerer declines simulcast; answer the way an SFU
        // (mediasoup) does, with the rids as recv, so the sender runs every layer.
        let answer =
            if layers.len() > 1 { simulcast_answer(&answer.to_string(), layers) } else { answer };
        sender.set_remote_description(answer).await.unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while sender.connection_state() != PeerConnectionState::Connected {
            assert!(Instant::now() < deadline, "loopback did not connect");
            let for_receiver = std::mem::take(&mut *to_receiver.lock().unwrap());
            for c in for_receiver {
                receiver.add_ice_candidate(c).await.unwrap();
            }
            let for_sender = std::mem::take(&mut *to_sender.lock().unwrap());
            for c in for_sender {
                sender.add_ice_candidate(c).await.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // Start the bandwidth estimate at the ladder's sum so every layer is
        // allocated at once instead of after a slow ramp on loopback.
        let total: u64 = layers.iter().map(|l| l.max_bitrate).sum();
        sender
            .set_bitrate(BitrateSettings {
                min_bitrate_bps: None,
                start_bitrate_bps: Some(total),
                max_bitrate_bps: Some(total * 2),
            })
            .unwrap();
        Pair { sender, receiver, source }
    }

    fn simulcast_answer(sdp: &str, layers: &[Layer]) -> SessionDescription {
        let mut out = String::new();
        for line in sdp.lines().filter(|l| !l.is_empty()) {
            out.push_str(line);
            out.push_str("\r\n");
            if line.starts_with("a=mid:") {
                for l in layers {
                    out.push_str(&format!("a=rid:{} recv\r\n", l.rid));
                }
                let rids: Vec<&str> = layers.iter().map(|l| l.rid).collect();
                out.push_str(&format!("a=simulcast:recv {}\r\n", rids.join(";")));
            }
        }
        SessionDescription::parse(&out, SdpType::Answer).expect("munged answer parses")
    }

    /// Feeds 30 fps frames from `src` into every source until `stop` is set.
    pub fn pump(
        src: Arc<ScreenSource>,
        sources: Vec<NativeVideoSource>,
        stop: Arc<AtomicBool>,
    ) -> std::thread::JoinHandle<u32> {
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut n = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let mut buf = I420Buffer::new(src.width as u32, src.height as u32);
                let (sy, su, sv) = buf.strides();
                let (y, u, v) = buf.data_mut();
                src.render(n, y, sy as usize, u, su as usize, v, sv as usize);
                let frame = VideoFrame {
                    rotation: VideoRotation::VideoRotation0,
                    timestamp_us: start.elapsed().as_micros() as i64,
                    frame_metadata: None,
                    buffer: buf,
                };
                for s in &sources {
                    s.capture_frame(&frame);
                }
                n += 1;
                let next = start + Duration::from_micros(n as u64 * 33_333);
                if let Some(wait) = next.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
            }
            n
        })
    }

    pub async fn outbound(pc: &PeerConnection) -> Vec<OutboundRtpStats> {
        let mut v: Vec<_> = pc
            .get_stats()
            .await
            .unwrap()
            .into_iter()
            .filter_map(|s| match s {
                RtcStats::OutboundRtp(o) if o.stream.kind != "audio" => Some(o),
                _ => None,
            })
            .collect();
        v.sort_by_key(|o| o.outbound.frame_width);
        v
    }

    pub fn print_outbound(label: &str, stats: &[OutboundRtpStats], seconds: f64) {
        println!("--- {label}");
        println!(
            "{:<4} {:>9} {:>6} {:>7} {:>5} {:>5} {:>8} {:>7} {:>8} {:>6} {:<6} {:<10} encoder",
            "rid",
            "size",
            "fps",
            "frames",
            "keys",
            "pli",
            "kbps",
            "avg_qp",
            "enc_ms",
            "pwr",
            "scal",
            "qlimit"
        );
        for o in stats {
            let s = &o.outbound;
            let frames = s.frames_encoded.max(1) as f64;
            println!(
                "{:<4} {:>9} {:>6.1} {:>7} {:>5} {:>5} {:>8.0} {:>7.1} {:>8.2} {:>6} {:<6} {:<10} {}",
                s.rid,
                format!("{}x{}", s.frame_width, s.frame_height),
                s.frames_per_second,
                s.frames_encoded,
                s.key_frames_encoded,
                s.pli_count + s.fir_count,
                o.sent.bytes_sent as f64 * 8.0 / 1000.0 / seconds,
                s.qp_sum as f64 / frames,
                s.total_encode_time * 1000.0 / frames,
                s.power_efficient_encoder,
                s.scalibility_mode,
                format!("{:?}", s.quality_limitation_reason),
                s.encoder_implementation,
            );
        }
    }

    /// Receives decoded frames on `pair.receiver`, pairs each with its source
    /// frame by barcode, and returns per-frame Y PSNR; writes PNG stills of the
    /// first decoded frame at or after each index in `stills`, with the
    /// matching source frame next to it.
    pub fn collect_quality(
        pair: &Pair,
        src: Arc<ScreenSource>,
        label: String,
        stills: Vec<u32>,
        warmup_frames: u32,
    ) -> (tokio::task::JoinHandle<()>, Arc<Mutex<Vec<(u32, f64)>>>) {
        let results: Arc<Mutex<Vec<(u32, f64)>>> = Default::default();
        let sink = results.clone();
        // on_track has already fired inside connect(); take the track from the
        // receiver directly.
        let track = pair.receiver.receivers().into_iter().find_map(|r| match r.track() {
            Some(MediaStreamTrack::Video(t)) => Some(t),
            _ => None,
        });
        let handle = tokio::spawn(async move {
            let Some(track) = track else { return };
            let mut stream = NativeVideoStream::new(track);
            let dir = out_dir();
            let (w, h) = (src.width, src.height);
            let (mut sy, mut su, mut sv) =
                (vec![0u8; w * h], vec![0u8; w * h / 4], vec![0u8; w * h / 4]);
            let mut pending = stills;
            let mut last = None;
            while let Some(frame) = stream.next().await {
                let i420 = frame.buffer.to_i420();
                if i420.width() as usize != w || i420.height() as usize != h {
                    continue;
                }
                let (dy, du, dv) = i420.data();
                let (ky, ku, kv) = i420.strides();
                let Some(n) = barcode::read(dy, ky as usize) else { continue };
                if last == Some(n) {
                    continue;
                }
                last = Some(n);
                src.render(n, &mut sy, w, &mut su, w / 2, &mut sv, w / 2);
                if n >= warmup_frames {
                    sink.lock().unwrap().push((n, psnr(&sy, w, dy, ky as usize, w, h)));
                }
                if let Some(pos) = pending.iter().position(|&p| n >= p) {
                    let p = pending.remove(pos);
                    write_png_i420(
                        &dir.join(format!("{label}_frame{p:05}_got{n:05}.png")),
                        w,
                        h,
                        (dy, ky as usize, du, ku as usize, dv, kv as usize),
                    );
                    let src_path = dir.join(format!("source_frame{n:05}.png"));
                    if !src_path.exists() {
                        write_png_i420(&src_path, w, h, (&sy, w, &su, w / 2, &sv, w / 2));
                    }
                }
            }
        });
        (handle, results)
    }
}

/// Three-layer simulcast with L1T3 on every encoding, H264 preferred, through
/// the real encoder factory. Answers: does each layer get its own hardware
/// encoder session, and is L1T3 honoured per layer. Rounds of 1, 2, 4 and 5
/// concurrent senders (3, 6, 12 and 15 encoder sessions) show where the
/// hardware runs out and what the factory does then. Override the rounds with
/// MF_SPIKE_SENDERS=1,2 and the duration with MF_SPIKE_SECONDS.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the native libwebrtc and a hardware H264 encoder MFT"]
async fn simulcast_l1t3_on_hardware_h264() {
    use rig::*;
    use std::sync::{atomic::Ordering, Arc};
    init_logging();
    let factory = libwebrtc::peer_connection_factory::PeerConnectionFactory::default();
    let (w, h) = (1920u32, 1080u32);
    let src = Arc::new(ScreenSource::new(w as usize, h as usize));
    let rounds: Vec<usize> = std::env::var("MF_SPIKE_SENDERS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 5]);
    let seconds: f64 =
        std::env::var("MF_SPIKE_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(12.0);
    for senders in rounds {
        take_lines();
        let mut pairs = Vec::new();
        for _ in 0..senders {
            pairs.push(connect(&factory, Codec::H264, &LADDER, w, h, "L1T3").await);
        }
        let stop = Arc::new(Flag::new(false));
        let pumper =
            pump(src.clone(), pairs.iter().map(|p| p.source.clone()).collect(), stop.clone());
        tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
        let mut all = Vec::new();
        for (i, p) in pairs.iter().enumerate() {
            let stats = outbound(&p.sender).await;
            print_outbound(&format!("{senders} sender(s), sender {i}"), &stats, seconds);
            all.extend(stats);
        }
        stop.store(true, Ordering::Relaxed);
        let frames = pumper.join().unwrap();
        for p in &pairs {
            p.sender.close();
            p.receiver.close();
        }
        drop(pairs);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let lines = take_lines();
        let mf = all
            .iter()
            .filter(|o| o.outbound.encoder_implementation.contains("MediaFoundation"))
            .count();
        let l1t3 = all.iter().filter(|o| o.outbound.scalibility_mode == "L1T3").count();
        println!(
            "== {senders} sender(s): {frames} source frames, {} layers, {mf} on MediaFoundation, \
             {l1t3} report L1T3",
            all.len()
        );
        for s in temporal_summaries(&lines) {
            println!("   temporal ids {s:?}");
        }
        print_lines(&lines, 40);
    }
}

/// Same screen content through H264 (MF) and VP8 at the same bitrate caps, a
/// single 1080p encoding with L1T3, decoded on the receiving peer. Writes
/// stills next to the matching source frame and prints Y PSNR. The numbers are
/// a guide; the judgement is made by looking at the stills.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the native libwebrtc and a hardware H264 encoder MFT"]
async fn screen_quality_samples() {
    use rig::*;
    use std::sync::{atomic::Ordering, Arc};
    init_logging();
    let factory = libwebrtc::peer_connection_factory::PeerConnectionFactory::default();
    let (w, h) = (1920u32, 1080u32);
    let src = Arc::new(ScreenSource::new(w as usize, h as usize));
    let seconds: f64 =
        std::env::var("MF_SPIKE_SECONDS").ok().and_then(|s| s.parse().ok()).unwrap_or(15.0);
    for bitrate in [4_000_000u64, 1_200_000] {
        for codec in [Codec::H264, Codec::Vp8] {
            take_lines();
            let label = format!("{codec:?}_{}k", bitrate / 1000).to_lowercase();
            let layer = [Layer { rid: "f", scale: 1.0, max_bitrate: bitrate }];
            let pair = connect(&factory, codec, &layer, w, h, "L1T3").await;
            // Stills: a static moment, two frames after a scroll, and mid-run.
            let collector =
                collect_quality(&pair, src.clone(), label.clone(), vec![150, 227, 360], 90);
            let stop = Arc::new(Flag::new(false));
            let pumper = pump(src.clone(), vec![pair.source.clone()], stop.clone());
            tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
            let stats = outbound(&pair.sender).await;
            print_outbound(&label, &stats, seconds);
            stop.store(true, Ordering::Relaxed);
            pumper.join().unwrap();
            pair.sender.close();
            pair.receiver.close();
            drop(pair);
            let (task, results) = collector;
            task.abort();
            let samples = std::mem::take(&mut *results.lock().unwrap());
            let finite: Vec<f64> = samples.iter().map(|s| s.1).filter(|p| p.is_finite()).collect();
            let mean = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
            let min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
            let after_scroll: Vec<f64> =
                samples.iter().filter(|s| s.0 % 15 < 3 && s.1.is_finite()).map(|s| s.1).collect();
            let scroll_mean = after_scroll.iter().sum::<f64>() / after_scroll.len().max(1) as f64;
            println!(
                "== {label}: {} decoded frames compared, Y PSNR mean {mean:.2} dB, min {min:.2} dB, \
                 first 3 frames after each scroll {scroll_mean:.2} dB",
                samples.len()
            );
            print_lines(&take_lines(), 12);
        }
    }
    println!("stills in {}", rig::out_dir().display());
}
