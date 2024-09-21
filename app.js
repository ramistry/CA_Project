const express = require('express');
const path = require('path');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;


app.use(express.static('public'));
app.use(bodyParser.json());


app.post('/chat', async (req, res) => {
    try {
        const userMessage = req.body.message;
        const response = await axios.post('http://127.0.0.1:8000/generate', { query: userMessage });
        res.json({ response: response.data.response });
    } catch (error) {
        res.status(500).json({ error: 'Error in chatbot API' });
    }
});


app.listen(port, () => {
    console.log(`App listening at http://localhost:${port}`);
});
