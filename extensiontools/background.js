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

// Handle right-click selection scraping
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "scrapeSelection" && info.selectionText) {
        let biasScore = Math.round(Math.random() * 100); // Convert to percentage
        let misinfoScore = Math.round(Math.random() * 100); // Convert to percentage
        let finalScore = Math.round((biasScore + misinfoScore) / 2); // Average Score for Highlighting

        console.log(`Simulated Scores - Bias: ${biasScore}%, Misinformation: ${misinfoScore}%, Final: ${finalScore}%`);

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (selectedText, biasScore, misinfoScore, finalScore) => {
                function getHighlightColor(score) {
                    if (score <= 30) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                    if (score <= 60) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                    return "rgba(255, 99, 71, 0.6)"; // Red ❌
                }

                let color = getHighlightColor(finalScore);

                // Add scores to the highlighted text in a user-friendly format
                let scoredText = `(Bias ${biasScore}%, Misinformation ${misinfoScore}%) ${selectedText}`;

                let selection = window.getSelection();
                if (selection.rangeCount > 0) {
                    let range = selection.getRangeAt(0);

                    let span = document.createElement("span");
                    span.className = "highlighted";
                    span.style.backgroundColor = color;
                    span.style.color = "inherit";
                    span.style.fontSize = "inherit";
                    span.style.lineHeight = "inherit";
                    span.style.padding = "2px";
                    span.style.borderRadius = "3px";
                    span.textContent = scoredText;

                    range.deleteContents();
                    range.insertNode(span);
                }
            },
            args: [info.selectionText, biasScore, misinfoScore, finalScore] // Ensure only serializable values
        });
    }
});

// Handle full-page scraping and highlighting
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "highlight_full") {
        let biasScore = Math.round(Math.random() * 100); // Convert to percentage
        let misinfoScore = Math.round(Math.random() * 100); // Convert to percentage
        let finalScore = Math.round((biasScore + misinfoScore) / 2); // Average Score for Highlighting

        console.log("=== Scraped Full Article JSON ===");
        console.log(JSON.stringify(request.data, null, 2));
        console.log(`Simulated Scores - Bias: ${biasScore}%, Misinformation: ${misinfoScore}%, Final: ${finalScore}%`);
        console.log("================================");

        chrome.scripting.executeScript({
            target: { tabId: sender.tab.id },
            func: (biasScore, misinfoScore, finalScore) => {
                function getHighlightColor(score) {
                    if (score <= 30) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                    if (score <= 60) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                    return "rgba(255, 99, 71, 0.6)"; // Red ❌
                }

                let color = getHighlightColor(finalScore);
                console.log(`Highlighting full article with color: ${color}`);

                let paragraphs = document.querySelectorAll("p");
                if (paragraphs.length > 0) {
                    // Add Bias & Misinformation Score to the FIRST paragraph only
                    let firstParagraph = paragraphs[0];
                    let span = document.createElement("span");
                    span.className = "highlighted";
                    span.style.backgroundColor = color;
                    span.style.color = "inherit";
                    span.style.fontSize = "inherit";
                    span.style.lineHeight = "inherit";
                    span.style.padding = "2px";
                    span.style.borderRadius = "3px";
                    span.innerHTML = `(Bias ${biasScore}%, Misinformation ${misinfoScore}%) ` + firstParagraph.innerHTML;
                    firstParagraph.innerHTML = "";
                    firstParagraph.appendChild(span);
                }

                // Apply highlight to remaining paragraphs (without scores)
                for (let i = 1; i < paragraphs.length; i++) {
                    let span = document.createElement("span");
                    span.className = "highlighted";
                    span.style.backgroundColor = color;
                    span.style.color = "inherit";
                    span.style.fontSize = "inherit";
                    span.style.lineHeight = "inherit";
                    span.style.padding = "2px";
                    span.style.borderRadius = "3px";
                    span.innerHTML = paragraphs[i].innerHTML;
                    paragraphs[i].innerHTML = "";
                    paragraphs[i].appendChild(span);
                }
            },
            args: [biasScore, misinfoScore, finalScore] // Pass only serializable values
        });
    }
});
