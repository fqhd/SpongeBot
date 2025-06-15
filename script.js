const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const chatBox = document.getElementById("chat-box");
let session;
let index_to_token;
let vocab;
let glblHiddn;

function appendMessage(text, sender = "user") {
	const message = document.createElement("div");
	message.classList.add("message");

	if (sender === "user") {
		message.classList.add("user-message");
	}

	message.textContent = text;
	chatBox.appendChild(message);
	chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
	const text = input.value.trim();
	if (text === "") return;

	// Add user's message on the right
	appendMessage(text, "user");

	// Simulate response after a short delay
	setTimeout(async () => {
		let sentence;
		[sentence, glblHiddn] = await generate(`Patrick: ${text}\n SpongeBob: `, 100, glblHiddn);
		appendMessage(sentence, "bot");
	}, 500);

	input.value = "";
}

(async () => {
	session = await ort.InferenceSession.create('spongebob_lstm_with_hidden.onnx');

	vocab = await fetch('vocab.json');
	vocab = await vocab.json();

	index_to_token = await fetch('index_to_token.json');
	index_to_token = await index_to_token.json();

	sendButton.addEventListener("click", sendMessage);
	input.addEventListener("keydown", (e) => {
		if (e.key === "Enter") sendMessage();
	});
})();

function tokensToIndices(tokens) {
	return tokens.map(token => vocab[token] !== undefined ? vocab[token] : 0);
}

function cleanAscii(text) {
	return [...text].filter(c => c.charCodeAt(0) < 128).join('');
}

function tokenize(text) {
	const tokens = [];
	let current = "";

	for (let i = 0; i < text.length; i++) {
		const char = text[i];

		if (/[a-zA-Z]/.test(char)) {
			current += char;
		} else {
			if (current) {
				tokens.push(current);
				current = "";
			}

			if (char === '\n') {
				tokens.push('\n');
			} else if (/\s/.test(char)) {
				tokens.push(' ');
			} else {
				tokens.push(char);
			}
		}
	}

	if (current) {
		tokens.push(current);
	}

	return tokens;
}

function softmax(arr) {
	const maxVal = Math.max(...arr); // for numerical stability
	const exps = arr.map(x => Math.exp(x - maxVal));
	const sumExps = exps.reduce((a, b) => a + b, 0);
	return exps.map(x => x / sumExps);
}

function sampleFromProbs(probs) {
	const r = Math.random();
	let cumulative = 0;
	for (let i = 0; i < probs.length; i++) {
		cumulative += probs[i];
		if (r < cumulative) return i;
	}
	return probs.length - 1; // fallback in case of floating point error
}

async function generate(input_sequence, maxLength, hidden) {
	let sentence = '';

	const tokens = tokenize(cleanAscii(input_sequence.toLowerCase()));

	const indices = tokensToIndices(tokens);

	let h_0;
	let c_0;
	if (!hidden) {
		const h0 = new Float32Array(512);
		const c0 = new Float32Array(512);
		h_0 = new ort.Tensor('float32', h0, [1, 1, 512]);
		c_0 = new ort.Tensor('float32', c0, [1, 1, 512]);
	} else {
		h_0 = hidden[0];
		c_0 = hidden[1];
	}

	for (const idx of indices.slice(0, -1)) {
		const feeds = {
			input_ids: new ort.Tensor('int64', new BigInt64Array([BigInt(idx)]), [1, 1]),
			h_0,
			c_0
		};

		const results = await session.run(feeds);

		h_0 = results.h_n;
		c_0 = results.c_n;
	}

	let inp = new ort.Tensor('int64', new BigInt64Array([BigInt(indices[indices.length - 1])]), [1, 1])

	for (let i = 0; i < maxLength; i++) {
		const feeds = {
			input_ids: inp,
			h_0,
			c_0
		};

		const results = await session.run(feeds);

		const logits = results.logits;
		h_0 = results.h_n;
		c_0 = results.c_n;

		logits.data[vocab['<UNK>']] = -Infinity;
        logits.data[vocab['[']] = -Infinity
		logits.data[vocab[']']] = -Infinity

		const probs = softmax(logits.data);

		const nextIdx = sampleFromProbs(probs);

		const nextToken = index_to_token[nextIdx.toString()];

		if (nextToken == '\n') {
			return [sentence, [h_0, c_0]];
		}
		
		sentence += nextToken;

		inp = new ort.Tensor('int64', new BigInt64Array([BigInt(nextIdx)]), [1, 1])
	}

	return [sentence, [h_0, c_0]];
}