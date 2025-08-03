const axios = require('axios');

async function sendMessage(message) {
  const resp = await axios.post(
    'https://api.openai.com/v1/chat/completions',
    {
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: message }]
    },
    { headers: { Authorization: `Bearer ${global.OPENAI_API_KEY}` } }
  );
  return resp.data.choices[0].message.content;
}

module.exports = { sendMessage };
