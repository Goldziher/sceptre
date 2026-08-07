//! Compute-device selection for the `candle` backend.
//!
//! Maps the backend-neutral [`Accelerator`] onto a [`Device`]. Per the `backend-seam`
//! decision this vocabulary stays inside `inference::`; the rest of the crate only ever
//! names an [`Accelerator`].
//!
//! Selection is deliberately loud, mirroring [`ort_ep`](super::super::ort_ep): an
//! explicit request that cannot be honored is an error, never a quiet fall back to the
//! CPU that would then be reported — and published as benchmark provenance — as a GPU
//! run. [`Accelerator::Auto`] is the one selection allowed to settle for the CPU, which
//! is what it asks for.

use candle_core::Device;

use super::candle_error;
use crate::config::Accelerator;
use crate::error::{OcrError, Result};

/// The device ordinal sceptre opens.
///
/// Multi-GPU selection is not exposed: it belongs on the configuration as a separate
/// index rather than smuggled into the accelerator's wire name.
#[cfg(any(feature = "candle-metal", feature = "candle-cuda"))]
const DEVICE_ORDINAL: usize = 0;

/// Open the device for `accelerator`, reporting which one was actually selected.
///
/// For [`Accelerator::Auto`] the answer may be [`Accelerator::Cpu`]; for every other
/// selection the returned accelerator is the requested one or the call fails.
pub(super) fn select_device(accelerator: Accelerator) -> Result<(Device, Accelerator)> {
    match accelerator {
        Accelerator::Cpu => Ok((Device::Cpu, Accelerator::Cpu)),
        Accelerator::Auto => Ok(preferred_device()),
        explicit => explicit_device(explicit),
    }
}

/// Walk the platform's preferred accelerators, keeping the first that opens.
fn preferred_device() -> (Device, Accelerator) {
    for &candidate in preferred_accelerators() {
        let Some(opened) = open(candidate) else {
            tracing::debug!(
                accelerator = candidate.as_str(),
                feature = cargo_feature(candidate),
                "skipping an accelerator this build was not compiled with"
            );
            continue;
        };
        match opened {
            Ok(device) => {
                tracing::info!(accelerator = candidate.as_str(), "opened the candle device");
                return (device, candidate);
            }
            Err(error) => tracing::warn!(
                accelerator = candidate.as_str(),
                %error,
                "the candle device could not be opened; trying the next candidate"
            ),
        }
    }
    tracing::info!(
        accelerator = "cpu",
        "no accelerator available; running candle on the CPU"
    );
    (Device::Cpu, Accelerator::Cpu)
}

/// Open a user-requested device, failing loudly if it cannot be used.
fn explicit_device(accelerator: Accelerator) -> Result<(Device, Accelerator)> {
    let Some(opened) = open(accelerator) else {
        return Err(OcrError::config(format!(
            "accelerator `{}` is not compiled into this build of sceptre; rebuild with the `{}` cargo feature",
            accelerator.as_str(),
            cargo_feature(accelerator)
        )));
    };
    let device = opened.map_err(|error| candle_error(&format!("open the `{}` device", accelerator.as_str()), error))?;
    tracing::info!(accelerator = accelerator.as_str(), "opened the candle device");
    Ok((device, accelerator))
}

/// Open the device for `accelerator`, or `None` when this build cannot address it.
///
/// The `None` answer covers both a missing cargo feature and an accelerator that is not
/// a candle device at all; configuration validation rejects the latter before it reaches
/// here, and the error above names a feature that does exist for everything it can see.
fn open(accelerator: Accelerator) -> Option<candle_core::Result<Device>> {
    match accelerator {
        Accelerator::Metal => metal_device(),
        Accelerator::Cuda => cuda_device(),
        Accelerator::Cpu | Accelerator::Auto | Accelerator::CoreMl | Accelerator::DirectMl => None,
    }
}

/// The accelerators [`Accelerator::Auto`] tries, most preferred first.
fn preferred_accelerators() -> &'static [Accelerator] {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        &[Accelerator::Metal]
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        &[Accelerator::Cuda]
    }
}

/// The sceptre cargo feature that compiles in support for `accelerator`.
fn cargo_feature(accelerator: Accelerator) -> &'static str {
    match accelerator {
        Accelerator::Metal => "candle-metal",
        Accelerator::Cuda => "candle-cuda",
        Accelerator::Cpu | Accelerator::Auto | Accelerator::CoreMl | Accelerator::DirectMl => "candle",
    }
}

#[cfg(feature = "candle-metal")]
fn metal_device() -> Option<candle_core::Result<Device>> {
    Some(Device::new_metal(DEVICE_ORDINAL))
}

#[cfg(not(feature = "candle-metal"))]
fn metal_device() -> Option<candle_core::Result<Device>> {
    None
}

#[cfg(feature = "candle-cuda")]
fn cuda_device() -> Option<candle_core::Result<Device>> {
    Some(Device::new_cuda(DEVICE_ORDINAL))
}

#[cfg(not(feature = "candle-cuda"))]
fn cuda_device() -> Option<candle_core::Result<Device>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_open_the_cpu_without_touching_a_device() {
        let (device, selected) = select_device(Accelerator::Cpu).expect("the cpu never fails");

        assert!(device.is_cpu());
        assert_eq!(selected, Accelerator::Cpu);
    }

    /// `Auto` must always resolve to something runnable, whatever the machine has.
    #[test]
    fn should_resolve_auto_to_a_concrete_device() {
        let (_device, selected) = select_device(Accelerator::Auto).expect("auto never fails");

        assert_ne!(selected, Accelerator::Auto, "auto must resolve to a real device");
        assert!(
            Accelerator::Cpu == selected || crate::config::Backend::Candle.supports(selected),
            "auto resolved to {selected:?}, which candle cannot run on"
        );
    }

    #[test]
    fn preferred_accelerators_never_list_cpu_or_auto() {
        let preferred = preferred_accelerators();
        assert!(!preferred.is_empty(), "every platform needs at least one candidate");
        assert!(
            preferred.iter().all(|candidate| crate::config::Backend::Candle
                .hardware_accelerators()
                .contains(candidate)),
            "every candidate must be an accelerator candle claims to support: {preferred:?}"
        );
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn apple_platforms_prefer_metal() {
        assert_eq!(preferred_accelerators(), &[Accelerator::Metal]);
    }

    #[test]
    fn every_candle_device_names_the_cargo_feature_that_enables_it() {
        assert_eq!(cargo_feature(Accelerator::Metal), "candle-metal");
        assert_eq!(cargo_feature(Accelerator::Cuda), "candle-cuda");
    }

    /// An accelerator that belongs to another backend is not a candle device.
    #[test]
    fn should_not_open_a_device_for_another_backends_accelerator() {
        assert!(open(Accelerator::CoreMl).is_none());
        assert!(open(Accelerator::DirectMl).is_none());
        assert!(open(Accelerator::Cpu).is_none());
    }

    /// A request this build cannot compile must name the feature to rebuild with,
    /// never silently run on the CPU.
    #[cfg(not(feature = "candle-cuda"))]
    #[test]
    fn should_reject_a_device_that_is_not_compiled_in() {
        let Err(error) = select_device(Accelerator::Cuda) else {
            panic!("cuda is not compiled in and must be rejected");
        };

        let message = error.to_string();
        assert!(message.contains("cuda"), "message must name the accelerator: {message}");
        assert!(
            message.contains("candle-cuda"),
            "message must name the feature: {message}"
        );
    }

    #[cfg(feature = "candle-metal")]
    #[test]
    fn should_open_metal_when_compiled_in() {
        let (device, selected) = select_device(Accelerator::Metal).expect("metal must open on this machine");

        assert_eq!(selected, Accelerator::Metal);
        assert!(device.is_metal());
    }
}
