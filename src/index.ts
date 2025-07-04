import { InferenceClient } from '@huggingface/inference';

const inference = new InferenceClient(process.env.HF_TOKEN_FINE_GRAINED);

async function embed() {
	const output = await inference.featureExtraction({
		inputs: 'My Cool Embeddings',
		model: 'BAAI/bge-small-zh-v1.5',
	});

	console.log(output);
}

embed();
