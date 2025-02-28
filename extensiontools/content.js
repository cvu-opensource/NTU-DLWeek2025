function extractArticleText() {
    let paragraphs = document.querySelectorAll("p");
    let articleText = Array.from(paragraphs).map(p => p.innerText).join(" ");

    let jsonData = JSON.stringify({ text: articleText });

    chrome.runtime.sendMessage({ action: "send_json", data: jsonData });
}

extractArticleText();
