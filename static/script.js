document.addEventListener("DOMContentLoaded", function () {
    const chatContainer = document.getElementById("chat-container");
    const userMessageInput = document.getElementById("user-message");
    const sendButton = document.getElementById("send-button");

    sendButton.addEventListener("click", async function () {
        await handleUserInput();
    });

    userMessageInput.addEventListener("keypress", async function (event) {
        if (event.key === "Enter") {
            await handleUserInput();
        }
    });

    // Handle user input
    async function handleUserInput() {
        const userMessage = userMessageInput.value.trim();
        if (userMessage !== "") {
            displayUserMessage(userMessage);
            try {
                await sendUserInput(userMessage);
            } catch (error) {
                console.error('Error:', error.message);
                displayChatbotMessage("An error occurred while processing your request.");
            }
            userMessageInput.value = "";
        }
    }
async function sendUserInput(userMessage) {
    try {
        const response = await fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'user_message=' + encodeURIComponent(userMessage),
        });

        if (!response.ok) {
            // Handle non-OK responses here
            console.error(`Server returned ${response.status}: ${response.statusText}`);
            displayChatbotMessage("Unable to fetch events. Please try again later.");
            return;
        }

        const responseData = await response.json();

        if (responseData.hasOwnProperty('events') && Array.isArray(responseData.events)) {
            displayEventsAsTable(responseData.events);
        } else {
            displayChatbotMessage(responseData.response);
        }
    } catch (error) {
        console.error('Error:', error);
        displayChatbotMessage("An unexpected error occurred while processing your request.");
    }
}

    // Display user message in the chat
    function displayUserMessage(message) {
        const userMessageElement = document.createElement("div");
        userMessageElement.className = "user-message";
        userMessageElement.textContent = message;
        chatContainer.appendChild(userMessageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Display chatbot response in the chat
    function displayChatbotMessage(message) {
        const chatbotMessageElement = document.createElement("div");
        chatbotMessageElement.className = "chatbot-message";
        chatbotMessageElement.textContent = message;
        chatContainer.appendChild(chatbotMessageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});

function displayEventsAsTable(events) {
    console.log("Displaying events as table", events); // Log the events data for debugging

    const table = document.createElement('table');
    table.className = 'events-table';
    table.style.width = '100%'; // Set the table width to full container width

    // Add table headers
    const headers = ['Event Name', 'City', 'Genre', 'Price'];
    const headerRow = table.insertRow();
    headers.forEach(headerText => {
        let header = headerRow.insertCell();
        header.textContent = headerText;
    });

    // Add event rows
    events.forEach(event => {
        const row = table.insertRow();
        headers.forEach(header => {
            let cell = row.insertCell();
            // Adjust the key to match your event data structure
            let value = event[header.replace(/ /g, '').toLowerCase()];
            if (typeof value === 'undefined') {
                console.error(`Error: ${header} is undefined in event data`, event);
                value = 'N/A';
            }
            cell.textContent = value;
        });
    });

    // Append the table to your chat container
    const chatContainer = document.getElementById('chat-container');
    chatContainer.appendChild(table);
    chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to the bottom of the chat
}
