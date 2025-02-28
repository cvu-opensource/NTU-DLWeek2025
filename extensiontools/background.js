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

        // Execute script in tab to get the article title
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (selectedText) => {
                let title = document.title || "Untitled Article";
                let jsonData = {
                    title: title,
                    text: selectedText
                };

                console.log("Selected Text Scraped (JSON):", JSON.stringify(jsonData, null, 2));
            },
            args: [info.selectionText]
        });
    }
});
