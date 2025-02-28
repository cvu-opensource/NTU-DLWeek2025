document.getElementById("scrape").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: extractArticleText
        });
    });
});

function extractArticleText() {
    let paragraphs = document.querySelectorAll("p");
    let articleText = Array.from(paragraphs).map(p => p.innerText).join(" ");
    let jsonData = JSON.stringify({ text: articleText });

    console.log("Extracted text:", articleText);
    chrome.runtime.sendMessage({ action: "send_json", data: jsonData });
}
