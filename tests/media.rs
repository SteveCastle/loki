use loki_reshoot::media::*;
use std::path::Path;

#[test]
fn ffmpeg_roundtrip() {
    let ff = Ffmpeg::discover(None).expect("ffmpeg");
    let dir = std::env::temp_dir().join("h3_media_test");
    std::fs::create_dir_all(&dir).unwrap();
    // synthetic 1 s clip + tone -> mp4
    let (w, h) = (64usize, 32usize);
    let wav: Vec<f32> = (0..2 * 32000).map(|i| ((i % 32000) as f32 * 0.05).sin() * 0.3).collect();
    let wavp = dir.join("a.wav");
    write_wav_f32(&wavp, &wav, 32000, 32000).unwrap();
    let outp = dir.join("t.mp4");
    let mut wr = VideoWriter::new(&ff, &outp, w, h, Some(&wavp), 20).unwrap();
    for f in 0..24u8 {
        let frame: Vec<u8> = (0..w * h * 3).map(|i| (i as u8).wrapping_add(f * 10)).collect();
        wr.write(&frame).unwrap();
    }
    wr.finish().unwrap();
    let info = ff.probe(&outp).unwrap();
    assert!(info.has_video && info.has_audio);
    assert_eq!((info.width, info.height), (w, h));
    let (rgb, n) = ff.decode_video(&outp, 0.0, None, 96, 48, 100).unwrap();
    assert_eq!(n, 24);
    assert_eq!(rgb.len(), 24 * 96 * 48 * 3);
    let (a, na) = ff.decode_audio(&outp, 0.0, Some(0.5), None).unwrap();
    assert!((na as i64 - 16000).abs() < 2000, "{na}");
    assert_eq!(a.len(), 2 * na);
    let (_, nb) = ff.decode_audio(&outp, 0.0, Some(0.5), Some(20000)).unwrap();
    assert_eq!(nb, 20000);
    let _ = Path::new("");
}
