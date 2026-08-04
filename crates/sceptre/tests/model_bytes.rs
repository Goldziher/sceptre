use std::sync::Arc;

use sceptre::{
    Language, ModelArtifact, ModelDescriptor, ModelProvider, ModelRole, OcrConfig, VerifiedModelProvider,
    model_descriptors,
};

const DETECTOR_SHA256: &str = "f2b3cbe41413047352141e5b863d87e696ec4f52b503040dba3a5700acd529a0";
const RECOGNIZER_SHA256: &str = "7c1243dcf122ad4912c3054d76fd28c84336d896272ce5b29e132fc4ba46a3df";

fn descriptor(role: ModelRole, name: &str, sha256: &str) -> ModelDescriptor {
    ModelDescriptor {
        name: name.to_string(),
        repo: format!("test/{name}"),
        revision: "main".to_string(),
        file: format!("{name}.onnx"),
        sha256: sha256.to_string(),
        role,
    }
}

#[test]
fn should_describe_models_without_inspecting_the_filesystem() {
    let mut config = OcrConfig::default();
    config.model.languages = vec![Language::English, Language::Telugu];
    config.model.cache_dir = Some("/a/path/that/need/not/exist".into());

    let descriptors = model_descriptors(&config).expect("descriptors only resolve registry metadata");

    assert_eq!(descriptors.len(), 3);
    assert_eq!(descriptors[0].role, ModelRole::Detector);
    assert_eq!(descriptors[0].file, "craft_mlt_25k.onnx");
    assert_eq!(descriptors[0].sha256.len(), 64);
    assert_eq!(descriptors[1].role, ModelRole::Recognizer(Language::English));
    assert_eq!(descriptors[2].role, ModelRole::Recognizer(Language::Telugu));
}

#[test]
fn should_serve_only_sha256_verified_model_bytes() {
    let detector = b"detector".to_vec();
    let recognizer = b"recognizer".to_vec();
    let provider = VerifiedModelProvider::new([
        (
            descriptor(ModelRole::Detector, "detector", DETECTOR_SHA256),
            Arc::<[u8]>::from(detector.clone()),
        ),
        (
            descriptor(
                ModelRole::Recognizer(Language::English),
                "recognizer",
                RECOGNIZER_SHA256,
            ),
            Arc::<[u8]>::from(recognizer.clone()),
        ),
    ])
    .expect("known hashes must verify");

    let ModelArtifact::Bytes(detector_bytes) = provider.detector().expect("detector bytes") else {
        panic!("verified in-memory provider must return bytes");
    };
    assert_eq!(detector_bytes.as_ref(), detector);
    let ModelArtifact::Bytes(recognizer_bytes) = provider.recognizer(Language::English).expect("recognizer bytes")
    else {
        panic!("verified in-memory provider must return bytes");
    };
    assert_eq!(recognizer_bytes.as_ref(), recognizer);
}

#[test]
fn should_reject_model_bytes_that_do_not_match_the_descriptor() {
    let error = VerifiedModelProvider::new([(
        descriptor(ModelRole::Detector, "detector", DETECTOR_SHA256),
        Arc::<[u8]>::from(&b"tampered"[..]),
    )])
    .expect_err("tampered bytes must fail verification");

    assert!(error.to_string().contains("sha256 mismatch"));
    assert!(error.to_string().contains("detector"));
}

#[test]
fn should_require_one_detector_and_requested_recognizer() {
    let provider = VerifiedModelProvider::new([(
        descriptor(ModelRole::Detector, "detector", DETECTOR_SHA256),
        Arc::<[u8]>::from(&b"detector"[..]),
    )])
    .expect("detector verifies");

    let error = provider
        .recognizer(Language::English)
        .expect_err("missing language must be explicit");

    assert!(error.to_string().contains("English"));
}
