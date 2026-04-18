const form = document.getElementById("uploadForm");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("https://space-backend-15tg.onrender.com/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    console.log(data);
    alert(data.message);
});