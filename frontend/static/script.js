// Chat functionality
const messagesContainer = document.getElementById('messagesContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Send message on button click
sendBtn.addEventListener('click', sendMessage);

// Send message on Enter key
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Clear input
    userInput.value = '';
    userInput.focus();
    
    // Disable send button
    sendBtn.disabled = true;
    
    // Add user message to DOM
    addMessage(message, 'user');
    
    // Scroll to bottom
    scrollToBottom();
    
    try {
        // Remove welcome message if it exists
        const welcomeMsg = messagesContainer.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }
        
        // Show loading indicator
        const loadingId = showLoading();
        
        // Send to backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        // Remove loading indicator
        removeLoading(loadingId);
        
        if (!response.ok) {
            if (response.status === 401) {
                addMessage('Session expired. Please log in again.', 'error');
                window.location.href = '/login';
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            addMessage(data.error, 'error');
        } else {
            addMessage(data.response, 'assistant');
        }
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, an error occurred. Please try again.', 'error');
    } finally {
        // Re-enable send button
        sendBtn.disabled = false;
        scrollToBottom();
    }
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    scrollToBottom();
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = `
        <div class="loading">
            <span></span><span></span><span></span>
            <span>Assistant is thinking...</span>
        </div>
    `;
    const loadingId = 'loading-' + Date.now();
    loadingDiv.id = loadingId;
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();
    return loadingId;
}

function removeLoading(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
