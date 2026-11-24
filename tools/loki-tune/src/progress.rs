//! Progress on stderr: bars on a terminal, nothing otherwise (the training loop prints parseable step lines).
use std::io::IsTerminal;

pub fn bar(quiet: bool, len: usize, label: &str) -> indicatif::ProgressBar {
    if quiet || !std::io::stderr().is_terminal() {
        return indicatif::ProgressBar::hidden();
    }
    let b = indicatif::ProgressBar::new(len as u64);
    b.set_style(indicatif::ProgressStyle::with_template(&format!("{label} {{bar:30}} {{pos}}/{{len}} {{eta}}")).unwrap());
    b
}

pub fn hms(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}
