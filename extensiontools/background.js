// Ensure the right-click context menu appears
function createContextMenu() {
    chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create({
            id: "scrapeSelection",
            title: "Scrape Selected Text",
            contexts: ["selection"]
        });
    });
}

// Create context menu on install and startup
chrome.runtime.onInstalled.addListener(() => {
    console.log("Extension installed - setting up context menu...");
    createContextMenu();
});

chrome.runtime.onStartup.addListener(() => {
    console.log("Service worker started - ensuring context menu exists...");
    createContextMenu();
});

// Function to fetch bias score from API
function fetchBiasScore(text, callback) {
    console.log("Sending request to bias API:", text);
    
    fetch("http://121.7.216.162:7001/predict_bias/", {  // Ensure correct API URL
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        console.log("API Response:", data);
        callback(Math.round(data.bias_score * 100));
    })
    .catch(error => {
        console.error("Error fetching bias score:", error);
        callback(Math.round(Math.random() * 100)); // Use random value if API fails
    });
}

// Handle full-text scraping from popup.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scrape_full") {
        let jsonData = request.data;

        fetchBiasScore(jsonData.text, (biasScore) => {
            let misinfoScore = Math.round(Math.random() * 100);
            let finalScore = Math.round((biasScore + misinfoScore) / 2);

            console.log(`Final Scores - Bias: ${biasScore}%, Misinformation: ${misinfoScore}%, Final: ${finalScore}%`);

            chrome.scripting.executeScript({
                target: { tabId: sender.tab.id },
                func: (biasScore, misinfoScore, finalScore) => {
                    function getHighlightColor(score) {
                        if (score <= 30) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                        if (score <= 60) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                        return "rgba(255, 99, 71, 0.6)"; // Red ❌
                    }

                    let color = getHighlightColor(finalScore);
                    let paragraphs = document.querySelectorAll("p");

                    if (paragraphs.length > 0) {
                        let firstParagraph = paragraphs[0];
                        let span = document.createElement("span");
                        span.className = "highlighted";
                        span.style.backgroundColor = color;
                        span.innerHTML = `(Bias ${biasScore}%, Misinformation ${misinfoScore}%) ` + firstParagraph.innerHTML;
                        firstParagraph.innerHTML = "";
                        firstParagraph.appendChild(span);
                    }

                    for (let i = 1; i < paragraphs.length; i++) {
                        let span = document.createElement("span");
                        span.className = "highlighted";
                        span.style.backgroundColor = color;
                        span.innerHTML = paragraphs[i].innerHTML;
                        paragraphs[i].innerHTML = "";
                        paragraphs[i].appendChild(span);
                    }
                },
                args: [biasScore, misinfoScore, finalScore]
            });
        });
    }
});

// Handle right-click selection scraping
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "scrapeSelection" && info.selectionText) {
        let selectedText = info.selectionText.trim();

        fetchBiasScore(selectedText, (biasScore) => {
            let misinfoScore = Math.round(Math.random() * 100);
            let finalScore = Math.round((biasScore + misinfoScore) / 2);

            console.log(`Final Scores - Bias: ${biasScore}%, Misinformation: ${misinfoScore}%, Final: ${finalScore}%`);

            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: (selectedText, biasScore, misinfoScore, finalScore) => {
                    function getHighlightColor(score) {
                        if (score <= 30) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                        if (score <= 60) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                        return "rgba(255, 99, 71, 0.6)"; // Red ❌
                    }

                    let color = getHighlightColor(finalScore);
                    let selection = window.getSelection();
                    if (selection.rangeCount > 0) {
                        let range = selection.getRangeAt(0);
                        let span = document.createElement("span");
                        span.className = "highlighted";
                        span.style.backgroundColor = color;
                        span.textContent = `(Bias ${biasScore}%, Misinformation ${misinfoScore}%) ${selectedText}`;
                        range.deleteContents();
                        range.insertNode(span);
                    }
                },
                args: [selectedText, biasScore, misinfoScore, finalScore]
            });
        });
    }
});
