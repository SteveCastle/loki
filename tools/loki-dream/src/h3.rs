//! The MiniMax H3 Image run: loki-reshoot's H3 engine (`loki_reshoot::h3image`) driven with this tool's model files,
//! residency cache, cancellation and progress, and its output delivered like every other architecture's.
//!
//! Text to image, multi-reference generation (`--ref`, the prompt's `<Picture n>`) and img2img (`--init` /
//! `--strength`). The model is distilled to 6 steps at CFG 1 on its own sigma grid and draws at least 4 MP (smaller
//! requests are generated at 4 MP and resampled, like the official pipeline).
use crate::cli::{self, Cli, Plan};
use crate::genmeta;
use crate::image::Rgb8;
use crate::models;
use anyhow::Result;
use loki_reshoot::h3image::{self, Host, Paths, Request};
use std::any::Any;
use std::path::Path;
use std::rc::Rc;

/// This tool's worker cache, cancel flag and progress output as the H3 engine's host.
struct DreamHost {
    bar: indicatif::ProgressBar,
    show_bar: bool,
}

impl Host for DreamHost {
    fn load_any(&self, key: &str, file: &Path, load: &mut dyn FnMut() -> Result<Rc<dyn Any>>) -> Result<Rc<dyn Any>> {
        // the cache stores the engine's model behind one more Rc
        crate::worker::load(key, file, || load()).map(|r: Rc<Rc<dyn Any>>| (*r).clone())
    }
    fn check_cancel(&self) -> Result<()> {
        crate::worker::check_cancel()
    }
    fn progress(&self, i: usize, n: usize) {
        self.bar.set_position(i as u64);
        if !self.show_bar {
            crate::info!("step {i}/{n}"); // machine-readable progress for callers that capture stderr
        }
    }
}

fn paths(cli: &Cli, dit: &Path) -> Result<Paths> {
    Ok(Paths {
        dit: dit.to_path_buf(),
        text_encoder: models::resolve(cli.text_encoder.clone(), &models::H3_TE)?,
        vae: models::resolve(cli.vae.clone(), &models::H3_IMAGE_VAE)?,
    })
}

fn to_engine(i: &Rgb8) -> loki_reshoot::image::Rgb8 {
    loki_reshoot::image::Rgb8 { w: i.w, h: i.h, data: i.data.clone() }
}

/// Worker warm-up: fetch the files and load the text encoder, VAE and DiT.
pub(crate) fn warm(cli: &Cli, dit: &Path) -> Result<()> {
    loki_reshoot::log::set_quiet(cli.quiet);
    let p = paths(cli, dit)?;
    let dev = h3image::device(crate::worker::active())?;
    let host = DreamHost { bar: indicatif::ProgressBar::hidden(), show_bar: false };
    h3image::warm(&dev, &host, &p)
}

pub(crate) fn run(cli: &Cli, plan: &Plan) -> Result<()> {
    loki_reshoot::log::set_quiet(cli.quiet);
    let prog = plan.prog;
    let ((gw, gh), (ow, oh)) = (plan.gen, plan.out);
    let p = paths(cli, &plan.dit)?;
    crate::info!(
        "{prog}: {} {}, {} image(s), {}x{}{}, {} steps, {} reference picture(s)",
        h3image::MODEL_NAME,
        cli::file_stem(&plan.dit),
        plan.jobs.len(),
        ow,
        oh,
        if (gw, gh) != (ow, oh) { format!(" (generated at {gw}x{gh})") } else { String::new() },
        plan.steps,
        plan.refs.len()
    );
    crate::info!("  dit: {}\n  text encoder: {}\n  vae: {}", p.dit.display(), p.text_encoder.display(), p.vae.display());
    let dev = h3image::device(crate::worker::active())?;
    let refs: Vec<loki_reshoot::image::Rgb8> = plan.refs.iter().map(to_engine).collect();
    let init = plan.init_img.as_ref().map(|i| (to_engine(i), cli.strength as f64));
    let t_all = std::time::Instant::now();
    let reqs: Vec<Request> = plan
        .jobs
        .iter()
        .map(|job| Request { prompt: plan.prompts[job.prompt].clone(), images: refs.clone(), width: gw, height: gh, steps: plan.steps, seed: job.seed, init: init.clone() })
        .collect();
    // one load of each model for every image; the progress bar restarts per image
    let (bar, show_bar) = cli::step_progress(cli, plan.steps);
    let host = DreamHost { bar, show_bar };
    let mut results: Vec<serde_json::Value> = Vec::new();
    h3image::generate_many(&dev, &host, &p, &reqs, &mut |ji, out| {
        let job = &plan.jobs[ji];
        let mut rec = genmeta::Record::new(prog, env!("CARGO_PKG_VERSION"), h3image::MODEL_NAME);
        rec.set("base_model", h3image::MODEL_NAME)
            .set("architecture", plan.arch.id())
            .set("prompt", plan.prompts[job.prompt].clone())
            .set("seed", job.seed)
            .set("steps", plan.steps)
            .set("cfg", 1.0)
            .set("turbo", true)
            .set("sampler", "euler")
            .set("scheduler", "simple (PDD grid)")
            .set("precision", "int8")
            .set("text_encoder", p.text_encoder.file_name().map(|f| f.to_string_lossy().into_owned()))
            .set("vae", p.vae.file_name().map(|f| f.to_string_lossy().into_owned()));
        let img = Rgb8 { w: out.image.w, h: out.image.h, data: out.image.data };
        cli::deliver(cli, plan, job, img, &mut rec, &mut results)
    })?;
    host.bar.finish_and_clear();
    cli::finish_json(cli, plan, results, true);
    crate::info!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}
