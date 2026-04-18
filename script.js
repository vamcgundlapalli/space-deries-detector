const form = document.getElementById("uploadForm");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = document.getElementById("file").files[0];

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("https://space-backend-15tg.onrender.com/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    let output = "Detected:<br>";

    data.detections.forEach(d => {
        output += `${d.class} (${d.confidence})<br>`;
    });

    document.getElementById("result").innerHTML = output;
});