import { pipeline, round } from '@huggingface/transformers';
import { WaveFile } from 'wavefile';

function test() {
	const result = round(8.24135811254686, 2);

	console.log(result);
}

async function embedder() {
	const embedder = await pipeline('feature-extraction', 'mixedbread-ai/mxbai-embed-large-v1');

	const docs = [
		'Represent this sentence for searching relevant passages: A man is eating a piece of bread',
		'A man is eating food.',
		'A man is eating pasta.',
		'The girl is carrying a baby.',
		'A man is riding a horse.',
	];

	const result = await embedder(docs, {
		pooling: 'mean',
		normalize: true,
	});

	console.log(result);
}

async function generateText() {
	const generator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M');
	const result = await generator('Give me good books list related to human psychology.', {
		max_new_tokens: 200,
		temperature: 0.7,
		repetition_penalty: 2.0,
	});

	console.log(result);
}

async function speechRecognition() {
	let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small.en');
	let url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
	// Fetch audio and convert to buffer
	const response = await fetch(url);
	const arrayBuffer = await response.arrayBuffer();
	const buffer = Buffer.from(arrayBuffer);

	let wav = new WaveFile(buffer);
	wav.toBitDepth('32f'); // Pipeline expects input as a FloatArray
	wav.toSampleRate(16000); // Whisper expects audio with a sampling rate of 16000

	let audioData: any = wav.getSamples();
	if (Array.isArray(audioData)) {
		if (audioData.length > 1) {
			const SCALING_FACTOR = Math.sqrt(2);
			for (let i = 0; i < audioData[0].length; ++i) {
				audioData[0][i] = (SCALING_FACTOR * (audioData[0][i] + audioData[1][i])) / 2;
			}
		}
		audioData = audioData[0];
	}

	let start = performance.now();
	let output = await transcriber(audioData);
	let end = performance.now();
	console.log(`Execution duration: ${(end - start) / 1000} seconds`);
	console.log(output);
}

speechRecognition();
