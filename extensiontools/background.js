// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//     if (request.action === "send_json") {
//         fetch("http://127.0.0.1:5000/analyze", {  // Change to your API URL
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: request.data
//         })
//         .then(response => response.json())
//         .then(data => console.log("API Response:", data))
//         .catch(error => console.error("Error:", error));
//     }
// });

function downloadJsonFile(jsonData) {
    try {
        let blob = new Blob([jsonData], { type: "application/json" });
        let url = URL.createObjectURL(blob);
        let a = document.createElement("a");
        a.href = url;
        a.download = "scraped_article.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error("Download failed:", error);
    }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "send_json") {
        console.log("Received scraped text:", request.data);
        downloadJsonFile(request.data);
    }
});
