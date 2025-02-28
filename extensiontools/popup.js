document.getElementById("scrape").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: extractArticleText
        });
    });
});

function extractArticleText() {
    let title = document.querySelector('meta[property="og:title"]')?.content || document.title || "Untitled Article";
    let paragraphs = document.querySelectorAll("p");
    let articleText = Array.from(paragraphs).map(p => p.innerText).join(" ");

    // Extract website name dynamically (source)
    let source = document.querySelector('meta[property="og:site_name"]')?.content || 
                 document.querySelector('meta[name="application-name"]')?.content ||
                 document.querySelector('meta[property="og:brand"]')?.content ||
                 window.location.hostname.replace("www.", "") || 
                 "None";

    // Author Extraction (Checking multiple possible locations)
    let author = document.querySelector('meta[name="author"]')?.content ||
                 document.querySelector('meta[property="article:author"]')?.content ||
                 document.querySelector('meta[name="byline"]')?.content ||
                 document.querySelector('.author, .byline, .post-author, .writer, .entry-author, .article__author-name')?.innerText ||
                 document.querySelector('a[rel="author"]')?.innerText ||
                 "None";

    // Published Date Extraction (Checking multiple possible locations)
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
        text: articleText.trim() || "None",
        source: source.trim() || "None",
        author: author.trim() || "None",
        published_date: publishedDate.trim() || "None"
    };

    console.log("Full Article Scraped (JSON):", JSON.stringify(jsonData, null, 2));

    chrome.runtime.sendMessage({ action: "send_json", data: JSON.stringify(jsonData) });
}
