# PetroAgent

1. Copy `OPENAI_API_KEY=your_key_here` into `.env`
2. `docker-compose up --build -d`
3. Browse http://localhost:3000  
4. Upload a CSV → pick input/target → chat with the agent (Enter to send).


sudo docker compose down
sudo docker system prune -a --volumes
sudo docker compose up -d --build
