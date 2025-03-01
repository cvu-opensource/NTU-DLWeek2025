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
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (selectedText) => {
                function getHighlightColor(score) {
                    if (score <= 30) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                    if (score <= 60) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                    return "rgba(255, 99, 71, 0.6)"; // Red ❌
                }

                let biasScore = Math.round(Math.random() * 100); // Simulated Bias Score
                let misinfoScore = Math.round(Math.random() * 100); // Simulated Misinformation Score
                let finalScore = Math.round((biasScore + misinfoScore) / 2); // Average Score for Highlighting
                let color = getHighlightColor(finalScore);

                // Extract metadata inside page context
                let title = document.querySelector('meta[property="og:title"]')?.content || document.title || "Untitled Article";
                let source = document.querySelector('meta[property="og:site_name"]')?.content || 
                             document.querySelector('meta[name="application-name"]')?.content ||
                             document.querySelector('meta[property="og:brand"]')?.content ||
                             window.location.hostname.replace("www.", "") || 
                             "None";
                let author = document.querySelector('meta[name="author"]')?.content ||
                             document.querySelector('meta[property="article:author"]')?.content ||
                             document.querySelector('.author, .byline, .post-author, .writer, .entry-author, .article__author-name')?.innerText ||
                             document.querySelector('a[rel="author"]')?.innerText ||
                             "None";
                let publishedDate = document.querySelector('meta[property="article:published_time"]')?.content || 
                                    document.querySelector('meta[name="publish-date"]')?.content || 
                                    document.querySelector('meta[name="date"]')?.content ||
                                    document.querySelector('meta[itemprop="datePublished"]')?.content ||
                                    document.querySelector('time')?.getAttribute("datetime") || 
                                    document.querySelector('.pubdate, .date-posted, .post-date, .entry-date, .article-date')?.innerText ||
                                    "None";

                let jsonData = {
                    title: title.trim() || "None",
                    text: selectedText.trim(),
                    source: source.trim() || "None",
                    author: author.trim() || "None",
                    published_date: publishedDate.trim() || "None"
                };

                // Log structured JSON output
                console.log("=== Scraped Selected Text JSON ===");
                console.log(JSON.stringify(jsonData, null, 2));
                console.log("==================================");

                // Highlight the selected text
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
                    span.textContent = `(Bias ${biasScore}%, Misinformation ${misinfoScore}%) ${selectedText}`;

                    range.deleteContents();
                    range.insertNode(span);
                }
            },
            args: [info.selectionText] // Ensure only serializable values
        });
    }
});

// Handle full-page scraping and highlighting
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "highlight_full") {
        chrome.scripting.executeScript({
            target: { tabId: sender.tab.id },
            func: () => {
                function getHighlightColor(score) {
                    if (score <= 30) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                    if (score <= 60) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                    return "rgba(255, 99, 71, 0.6)"; // Red ❌
                }

                let biasScore = Math.round(Math.random() * 100); // Simulated Bias Score
                let misinfoScore = Math.round(Math.random() * 100); // Simulated Misinformation Score
                let finalScore = Math.round((biasScore + misinfoScore) / 2); // Average Score for Highlighting
                let color = getHighlightColor(finalScore);

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
            }
        });
    }
});
