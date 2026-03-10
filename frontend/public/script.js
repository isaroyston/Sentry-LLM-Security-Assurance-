document.getElementById('chat-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const userInput = document.getElementById('user-input');
    const userMessageText = userInput.value.trim();

    if (userMessageText === '') return;

    // Display user's message
    appendMessage('user', userMessageText);

    // Clear input
    userInput.value = '';

    // Node.js Backend Call (Placeholder)
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userMessageText })
        });

        const data = await response.json();
        appendMessage('assistant', data.response);
    } catch (error) {
        console.error('Error:', error);
        appendMessage('assistant', "Something went wrong. Ensure you've run 'npm start' in the frontend folder.");
    }
});

function appendMessage(sender, text) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.innerText = text;

    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
