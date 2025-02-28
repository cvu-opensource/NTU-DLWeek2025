function createContextMenu() {
    chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create({
            id: "scrapeSelection",
            title: "Scrape Selected Text",
            contexts: ["selection"]
        });
    });
}

chrome.runtime.onInstalled.addListener(() => {
    console.log("Extension installed - setting up context menu...");
    createContextMenu();
});

chrome.runtime.onStartup.addListener(() => {
    console.log("Service worker started - ensuring context menu exists...");
    createContextMenu();
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "scrapeSelection" && info.selectionText) {
        console.log("Selected text scraped (background service worker):", info.selectionText);

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (selectedText) => {
                let title = document.querySelector('meta[property="og:title"]')?.content || document.title || "Untitled Article";

                // Extract website name dynamically (source)
                let source = document.querySelector('meta[property="og:site_name"]')?.content || 
                             document.querySelector('meta[name="application-name"]')?.content ||
                             document.querySelector('meta[property="og:brand"]')?.content ||
                             window.location.hostname.replace("www.", "") || 
                             "None";

                // Author Extraction
                let author = document.querySelector('meta[name="author"]')?.content ||
                             document.querySelector('meta[property="article:author"]')?.content ||
                             document.querySelector('meta[name="byline"]')?.content ||
                             document.querySelector('.author, .byline, .post-author, .writer, .entry-author, .article__author-name')?.innerText ||
                             document.querySelector('a[rel="author"]')?.innerText ||
                             "None";

                // Published Date Extraction
                let publishedDate = document.querySelector('meta[property="article:published_time"]')?.content || 
                                    document.querySelector('meta[name="publish-date"]')?.content || 
                                    document.querySelector('meta[name="date"]')?.content ||
                                    document.querySelector('meta[name="dc.date"]')?.content ||
                                    document.querySelector('meta[itemprop="datePublished"]')?.content ||
                                    document.querySelector('time')?.getAttribute("datetime") || 
                                    document.querySelector('.pubdate, .date-posted, .post-date, .entry-date, .article-date')?.innerText ||
                                    "None";

                let jsonData = {
                    title: title.trim() || "None",
                    text: selectedText.trim() || "None",
                    source: source.trim() || "None",
                    author: author.trim() || "None",
                    published_date: publishedDate.trim() || "None"
                };

                console.log("Selected Text Scraped (JSON):", JSON.stringify(jsonData, null, 2));
            },
            args: [info.selectionText]
        });
    }
});
