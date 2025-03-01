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

// Handle full-text scraping from popup.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scrape_full") {
        let jsonData = request.data;

        // Fetch bias score from API
        fetch("http://localhost:5000/predict_bias/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: jsonData.text })
        })
        .then(response => response.json())
        .then(data => {
            let biasScore = Math.round(data.bias_score * 100);
            let misinfoScore = Math.round(Math.random() * 100); // Simulated Misinformation Score
            let finalScore = Math.round((biasScore + misinfoScore) / 2);

            console.log(`API Response - Bias: ${biasScore}%, Simulated Misinformation: ${misinfoScore}%, Final: ${finalScore}%`);

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
        })
        .catch(error => {
            console.error("Error fetching bias score:", error);
        });
    }
});

// Handle right-click selection scraping
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "scrapeSelection" && info.selectionText) {
        let selectedText = info.selectionText.trim();

        // Fetch bias score from API
        fetch("http://localhost:5000/predict_bias/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: selectedText })
        })
        .then(response => response.json())
        .then(data => {
            let biasScore = Math.round(data.bias_score * 100);
            let misinfoScore = Math.round(Math.random() * 100);
            let finalScore = Math.round((biasScore + misinfoScore) / 2);

            console.log(`API Response - Bias: ${biasScore}%, Simulated Misinformation: ${misinfoScore}%, Final: ${finalScore}%`);

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
        })
        .catch(error => {
            console.error("Error fetching bias score:", error);
        });
    }
});
