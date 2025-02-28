document.getElementById("scrape").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: extractArticleText
        });
    });
});

function extractArticleText() {
    let title = document.title || "Untitled Article"; // Extract page title
    let paragraphs = document.querySelectorAll("p");
    let articleText = Array.from(paragraphs).map(p => p.innerText).join(" ");

    let jsonData = {
        title: title,
        text: articleText
    };

    console.log("Full Article Scraped (JSON):", JSON.stringify(jsonData, null, 2));

    chrome.runtime.sendMessage({ action: "send_json", data: JSON.stringify(jsonData) });
}
