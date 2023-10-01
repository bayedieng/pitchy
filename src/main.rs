use std::thread;
use hound::WavReader;

use ndarray::{concatenate, prelude::*};
use ndarray_stats::QuantileExt;

use ort::environment::Environment;
use ort::session::SessionBuilder;
use ort::tensor::OrtOwnedTensor;

use ort::{OrtResult, Value};
use samplerate::{convert, ConverterType};

const MODEL_SAMPLE_RATE: u32 = 16000;
const WINDOW_SIZE: usize = 1024;
const CENTS_PER_BINS: f32 = 20.;

/// Equivalent of pytorch unfold function
fn im2col(input: &Array2<f32>, kernel_height: usize, kernel_width: usize, stride_height: usize, stride_width: usize) -> Array2<f32> {
    let (input_height, input_width) = input.dim();
    
    // Calculate the output dimensions
    let output_height = (input_height - kernel_height) / stride_height + 1;
    let output_width = (input_width - kernel_width) / stride_width + 1;

    // Create a new array to store the columns
    let mut cols = Array2::zeros((kernel_height * kernel_width, output_height * output_width));
    for i in 0..output_height {
        for j in 0..output_width {
            let start_row = i * stride_height;
            let start_col = j * stride_width;
            let end_row = start_row + kernel_height;
            let end_col = start_col + kernel_width;

            // Slice the input to get the current window
            let window = input.slice(s![
                start_row..end_row,
                start_col..end_col,
            ])
            .into_shape(1024)
            .unwrap();


            // Assign the window to the corresponding column
            cols.slice_mut(s![.., i * output_width + j]).assign(&window);

        }
    }
    cols
}


struct ModelInput {
    pub array: Array2<f32>,
    time_hop: usize
}

/// default settings of torchcrepe preprocess function
fn preprocess(audio_file_path: &str) -> ModelInput{
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

    let ret_array = im2col(&array_ndarray, 1, WINDOW_SIZE, 1, hop_length)
        .t()
        .to_owned();
    ModelInput { array: ret_array.clone(), time_hop: ret_array.view().shape()[0] }
}

fn cents_to_frequency(cents: f32) -> f32 {
    10. * 2.0f32.powf(cents / 1200.0)
}

fn bins_to_cents(bins: f32) -> f32 {
    CENTS_PER_BINS * bins + 1997.3794084376191
}

fn bins_to_frequency(bins: f32) -> f32 {
    cents_to_frequency(bins_to_cents(bins))
}

fn postprocess(logits: &mut Array2<f32>) -> f32 {
    let (_, bins) = logits.argmax().unwrap();
    bins_to_frequency(bins as f32)
    
}

fn main() -> OrtResult<()> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(ort::LoggingLevel::Verbose)
        .build()?
        .into_arc();

    let n_threads = thread::available_parallelism().unwrap().get() / 2;
 
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
        .with_intra_threads(n_threads as i16)?
        .with_model_from_file("crepe.onnx")?;

    let input_data = preprocess("c_note.wav");
    let cow_array = CowArray::from(input_data.array).into_dyn();
    let value = Value::from_array(session.allocator(), &cow_array)?;
    let infer = session.run(vec![value])?;
    
    let y: OrtOwnedTensor<f32, _> = infer[0].try_extract()?;
    let y_slice = y.view().as_slice().unwrap().to_vec();
    let mut post_array = Array2::from_shape_vec((input_data.time_hop, 360), y_slice).unwrap();
    let pitch = postprocess(&mut post_array);
    println!("{pitch}");
    
    Ok(())

}
