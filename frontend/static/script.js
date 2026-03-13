const chatForm = document.getElementById('chat-form');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const submitButton = chatForm.querySelector('button');

chatForm.addEventListener('submit', async function (e) {
    e.preventDefault();

    const userMessageText = userInput.value.trim();

    if (userMessageText === '') return;

    appendMessage('user', userMessageText);
    userInput.value = '';

    setPendingState(true);
    const loadingMessage = appendLoadingMessage();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userMessageText })
        });

        const data = await response.json();
        loadingMessage.remove();

        if (!response.ok) {
            appendMessage('assistant', data.error || 'Something went wrong while contacting the chatbot.');
            return;
        }

        appendMessage('assistant', data.response);
    } catch (error) {
        console.error('Error:', error);
        loadingMessage.remove();
        appendMessage('assistant', 'Something went wrong. Ensure the Flask app is running in the frontend folder.');
    } finally {
        setPendingState(false);
        userInput.focus();
    }
});

function appendMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.innerText = text;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function appendLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'assistant', 'loading-message');
    loadingDiv.setAttribute('aria-label', 'Assistant is generating a response');

    const label = document.createElement('span');
    label.classList.add('loading-label');
    label.innerText = 'Generating answer';

    const dots = document.createElement('span');
    dots.classList.add('loading-dots');
    dots.setAttribute('aria-hidden', 'true');

    for (let index = 0; index < 3; index += 1) {
        const dot = document.createElement('span');
        dot.classList.add('loading-dot');
        dots.appendChild(dot);
    }

    loadingDiv.append(label, dots);
    chatMessages.appendChild(loadingDiv);
    scrollToBottom();
    return loadingDiv;
}

function setPendingState(isPending) {
    userInput.disabled = isPending;
    submitButton.disabled = isPending;
    submitButton.innerText = isPending ? 'Waiting...' : 'Send';
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
