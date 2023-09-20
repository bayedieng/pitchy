use hound::WavReader;

use ndarray::{concatenate, prelude::*};

use ort::environment::Environment;
use ort::session::SessionBuilder;
use ort::tensor::OrtOwnedTensor;

use ort::{OrtResult, Value};
use samplerate::{convert, ConverterType};

const MODEL_SAMPLE_RATE: u32 = 16000;
const WINDOW_SIZE: usize = 1024;

/// Equivalent of pytorch unfold function
fn im2col(input: Array2<f32>, kernel_size: (usize, usize), stride: (usize, usize)) -> Array2<f32> {
    let (input_height, input_width) = input.dim();
    let (kernel_height, kernel_width) = kernel_size;
    let (stride_height, stride_width) = stride;
    let col_height = (input_height - kernel_height) / stride_height + 1;
    let col_width = (input_width - kernel_width) / stride_width + 1;

    let mut col_matrix = Array::zeros((kernel_height * kernel_width, col_height * col_width));

    for i in 0..col_height {
        for j in 0..col_width {
            let start_x = i * stride_height;
            let start_y = j * stride_width;
            let end_x = start_x + kernel_height;
            let end_y = start_y + kernel_width;

            let col_slice = col_matrix.slice_mut(s![.., i * col_width + j]).to_owned();
            let col_slice_len = col_slice.dim();
            let mut col_slice = col_slice.into_shape((1, col_slice_len)).unwrap();
            col_slice.assign(&input.slice(s![start_x..end_x, start_y..end_y]));
        }
    }
    col_matrix
}

/// default settings of torchcrepe preprocess function
fn preprocess(audio_file_path: &str) -> Array2<f32> {
    let reader = WavReader::open(audio_file_path).unwrap();
    let spec = reader.spec();
    let hop_length = MODEL_SAMPLE_RATE as usize / 100;
    let audio_samples: Vec<f32> = reader
        .into_samples::<i32>()
        .map(|s| s.unwrap() as f32)
        .collect();
    let resampled_audio = convert(
        spec.sample_rate,
        MODEL_SAMPLE_RATE,
        1,
        ConverterType::SincBestQuality,
        &audio_samples,
    )
    .unwrap();

    let audio_ndarray =
        Array2::from_shape_vec((1, resampled_audio.len()), resampled_audio).unwrap();
    let array_padder = Array2::<f32>::zeros((1, WINDOW_SIZE / 2));
    let array_ndarray = concatenate(
        Axis(1),
        &[
            audio_ndarray.view(),
            array_padder.view(),
            array_padder.view(),
        ],
    )
    .unwrap();

    im2col(array_ndarray, (1, WINDOW_SIZE), (1, hop_length))
        .t()
        .to_owned()
}

fn main() -> OrtResult<()> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(ort::LoggingLevel::Verbose)
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
        .with_intra_threads(4)?
        .with_model_from_file("crepe.onnx")?;

    let input_data = preprocess("c_note.wav");
    print!("{:?}", &input_data.shape());
    let cow_array = CowArray::from(input_data).into_dyn();
    let value = Value::from_array(session.allocator(), &cow_array)?;
    let infer = session.run(vec![value])?;
    let y: OrtOwnedTensor<f32, _> = infer[0].try_extract()?;

    Ok(())
}
