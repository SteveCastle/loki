//! The generation record every loki- tool stamps into what it writes (see `tools/CONVENTIONS.md`, "Provenance").
//!
//! One JSON object describing the run: which tool and model, the final prompt, the sampling settings, the inputs
//! and the command line. loki-retouch stores it in a PNG `iTXt` chunk with the keyword `loki`; loki-reshoot in the
//! mp4 container's `comment` tag. The Lowkey Media Viewer and Server read it back into their Generation panel
//! (`src/renderer/generation-metadata.ts`), next to the ComfyUI / WebUI records other tools leave behind.
//!
//! The record is recognised by its two marker fields: `"loki": <format version>` and `"tool": "<name>"`.
//! Fields are added, never renamed: a reader of version 1 must keep working on later records.
//!
//! `meta` is the caller's own JSON object (`--meta` / `--meta-file`), stored verbatim: an app's recipe for the run in
//! its own terms (the Lowkey Transform flow keeps the intent and settings there to rebuild a saved task from a result).
use serde_json::{Map, Value};

/// PNG text keyword and the JSON format version.
pub const PNG_KEYWORD: &str = "loki";
pub const FORMAT_VERSION: u64 = 1;

/// Builder for the record. Keys keep insertion order (serde_json's default map does not, so the order is
/// re-imposed on output: markers, tool identity, then everything else in the order it was set).
#[derive(Clone, Debug)]
pub struct Record {
    fields: Vec<(String, Value)>,
}

impl Record {
    /// `tool` is the executable name as the user knows it (`loki-retouch`), `model` the engine's model in words
    /// (`Qwen Image 2.1`). The command line (every argument after the program name) is captured here.
    pub fn new(tool: &str, version: &str, model: &str) -> Record {
        let mut r = Record { fields: Vec::new() };
        r.set("loki", FORMAT_VERSION);
        r.set("tool", tool);
        r.set("version", version);
        r.set("model", model);
        r.set("created", iso8601_now());
        r.set("args", command_line(std::env::args().skip(1)));
        r
    }

    /// Set (or replace) a field. `Value::Null` removes it, so optional settings can be passed straight through.
    pub fn set<V: Into<Value>>(&mut self, key: &str, value: V) -> &mut Record {
        let value = value.into();
        self.fields.retain(|(k, _)| k != key);
        if !value.is_null() {
            self.fields.push((key.to_string(), value));
        }
        self
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn to_value(&self) -> Value {
        let mut m = Map::with_capacity(self.fields.len());
        for (k, v) in &self.fields {
            m.insert(k.clone(), v.clone());
        }
        Value::Object(m)
    }

    /// Compact JSON, the form that goes into the file.
    pub fn to_json(&self) -> String {
        // serde_json's Map is a BTreeMap by default (keys sorted); build the string by hand to keep the order
        // meaningful to a human reading the raw chunk.
        let mut s = String::from("{");
        for (i, (k, v)) in self.fields.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&Value::String(k.clone()).to_string());
            s.push(':');
            s.push_str(&v.to_string());
        }
        s.push('}');
        s
    }
}

/// The command line for the record's `args`: as given, minus `--meta` / `--meta-file` (the object has its own field,
/// and a temp file's path means nothing later).
pub fn command_line<I: IntoIterator<Item = String>>(args: I) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_value = false;
    for a in args {
        if skip_value {
            skip_value = false;
        } else if a == "--meta" || a == "--meta-file" {
            skip_value = true;
        } else if !(a.starts_with("--meta=") || a.starts_with("--meta-file=")) {
            out.push(a);
        }
    }
    out
}

/// The `meta` object from `--meta JSON` or `--meta-file FILE` (at most one is given): it must be a JSON object, so a
/// mistake fails the run before any model loads instead of silently stamping nothing.
pub fn read_meta(inline: Option<&str>, file: Option<&str>) -> Result<Option<Value>, String> {
    let text = match (inline, file) {
        (Some(_), Some(_)) => return Err("use --meta or --meta-file, not both".into()),
        (Some(t), None) => t.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).map_err(|e| format!("reading --meta-file {f}: {e}"))?,
        (None, None) => return Ok(None),
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(v @ Value::Object(_)) => Ok(Some(v)),
        Ok(_) => Err("--meta must be a JSON object".into()),
        Err(e) => Err(format!("--meta is not valid JSON: {e}")),
    }
}

/// The current time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn iso8601_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    iso8601_utc(secs)
}

/// Unix seconds -> `YYYY-MM-DDTHH:MM:SSZ` (proleptic Gregorian; Howard Hinnant's civil_from_days).
pub fn iso8601_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(951_782_400), "2000-02-29T00:00:00Z"); // leap day
        assert_eq!(iso8601_utc(1_792_000_000), "2026-10-14T17:46:40Z");
    }

    #[test]
    fn record_shape() {
        let mut r = Record::new("loki-test", "9.9.9", "Test Model");
        r.set("prompt", "a prompt").set("seed", 7u64).set("preset", Value::Null).set("size", vec![16u64, 32]);
        let s = r.to_json();
        assert!(s.starts_with(r#"{"loki":1,"tool":"loki-test","version":"9.9.9","model":"Test Model","created":""#), "{s}");
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["prompt"], "a prompt");
        assert_eq!(v["seed"], 7);
        assert!(v.get("preset").is_none(), "null settings are left out");
        assert_eq!(v["size"], serde_json::json!([16, 32]));
        assert!(v["args"].is_array());
        assert_eq!(r.get("seed"), Some(&Value::from(7u64)));
        // replacing keeps one copy
        r.set("seed", 8u64);
        assert_eq!(r.to_value()["seed"], 8);
        assert_eq!(r.fields.iter().filter(|(k, _)| k == "seed").count(), 1);
    }

    #[test]
    fn meta_is_an_object() {
        assert_eq!(read_meta(None, None), Ok(None));
        let v = read_meta(Some(r#"{"kind":"x","v":1}"#), None).unwrap().unwrap();
        assert_eq!(v["kind"], "x");
        assert!(read_meta(Some("[1]"), None).is_err());
        assert!(read_meta(Some("{nope"), None).is_err());
        assert!(read_meta(Some("{}"), Some("f")).is_err());
        let dir = std::env::temp_dir().join(format!("genmeta-test-{}", std::process::id()));
        std::fs::write(&dir, r#"{"from":"file"}"#).unwrap();
        let v = read_meta(None, dir.to_str()).unwrap().unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!(v["from"], "file");
    }

    #[test]
    fn command_line_drops_meta() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            command_line(a(&["-p", "x", "--meta", "{}", "--meta-file", "/tmp/m.json", "--meta={}", "--meta-file=/t", "in.png"])),
            a(&["-p", "x", "in.png"])
        );
    }
}
