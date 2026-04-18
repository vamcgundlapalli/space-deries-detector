console.log("JS LOADED SUCCESSFULLY");

const form = document.getElementById("uploadForm");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("https://space-backend-15tg.onrender.com/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        document.getElementById("result").innerText = data.message;
    } catch (err) {
        console.error(err);
        alert("Error connecting to backend");
    }
});