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
        let fakeScore = parseFloat((Math.random()).toFixed(2)); // Simulated API score
        console.log("Simulated API Score (Selection):", fakeScore);

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (selectedText, score) => {
                function getHighlightColor(score) {
                    if (score <= 0.3) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                    if (score <= 0.6) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                    return "rgba(255, 99, 71, 0.6)"; // Red ❌
                }

                let color = getHighlightColor(score);

                // Extract metadata
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
                    span.textContent = selectedText;

                    range.deleteContents();
                    range.insertNode(span);
                }
            },
            args: [info.selectionText, fakeScore] // Ensure only strings & numbers are passed
        });
    }
});

// Handle full-page scraping and highlighting
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "highlight_full") {
        console.log("=== Scraped Full Article JSON ===");
        console.log(JSON.stringify(request.data, null, 2));
        console.log("================================");

        console.log("Applying highlight with score:", request.score);

        chrome.scripting.executeScript({
            target: { tabId: sender.tab.id },
            func: (score) => {
                function getHighlightColor(score) {
                    if (score <= 0.3) return "rgba(144, 238, 144, 0.6)"; // Light Green ✅
                    if (score <= 0.6) return "rgba(255, 255, 0, 0.6)"; // Yellow ⚠️
                    return "rgba(255, 99, 71, 0.6)"; // Red ❌
                }

                let color = getHighlightColor(score);
                console.log(`Highlighting full article with color: ${color}`);

                document.querySelectorAll("p").forEach(p => {
                    let span = document.createElement("span");
                    span.className = "highlighted";
                    span.style.backgroundColor = color;
                    span.style.color = "inherit";
                    span.style.fontSize = "inherit";
                    span.style.lineHeight = "inherit";
                    span.style.padding = "2px";
                    span.style.borderRadius = "3px";
                    span.innerHTML = p.innerHTML;
                    p.innerHTML = "";
                    p.appendChild(span);
                });
            },
            args: [request.score] // Pass only serializable values
        });
    }
});
