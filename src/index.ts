import { InferenceClient } from '@huggingface/inference';
import { writeFileSync } from 'fs';

const inference = new InferenceClient(process.env.HF_TOKEN_FINE_GRAINED);

async function embed() {
	const output = await inference.featureExtraction({
		inputs: 'My Cool Embeddings',
		model: 'BAAI/bge-small-zh-v1.5',
	});

	console.log(output);
}

async function translate() {
	const result = await inference.translation({
		inputs: 'How is the weather in Paris?',
		model: 'google-t5/t5-base',
	});

	console.log(result);
}

async function answerQuestion() {
	const result = await inference.questionAnswering({
		inputs: {
			context: 'The quick brown fox jumps over the lazy dog',
			question: 'What color is the dog?',
		},
		model: 'distilbert-base-cased-distilled-squad',
	});
	console.log(result);
}

async function textToImage() {
	const result = await inference.textToImage(
		{
			inputs: 'Dog in the hat on a mat.',
			model: 'black-forest-labs/FLUX.1-dev',
			parameters: {
				negative_prompt: 'blurry and unclear',
			},
		},
		{ outputType: 'blob' }
	);

	writeFileSync('image.png', Buffer.from(await result.arrayBuffer()));
	console.log('Image saved successfully');
}

// embed()
// translate();
// answerQuestion();
textToImage();
