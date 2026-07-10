# 🔗 URL Shortener API

A lightweight and efficient RESTful URL Shortener built using **Python**, **Flask**, and **SQLite**. This project allows users to shorten long URLs, redirect using short links, track click analytics, and manage shortened URLs through a clean REST API.

---

## 📌 Project Overview

Long URLs are difficult to share and manage. This project provides a simple backend service that converts long URLs into short, unique links while maintaining analytics such as click counts.

The application demonstrates the fundamentals of backend development including:

- REST API design
- Database integration
- CRUD operations
- Input validation
- URL redirection
- Analytics tracking

---

## ✨ Features

- 🔗 Shorten long URLs
- ✅ Validate URLs before shortening
- ♻ Prevent duplicate URL entries
- 🚀 Redirect users using short URLs
- 📊 Track click analytics
- 🗑 Delete shortened URLs
- 💾 Store data using SQLite
- 📦 JSON-based API responses
- ⚡ Lightweight Flask backend

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3 | Programming Language |
| Flask | Backend Framework |
| SQLite | Database |
| Validators | URL Validation |
| Thunder Client / Postman | API Testing |
| Git & GitHub | Version Control |

---

## 📂 Project Structure

```
url-shortener-api/
│
├── app.py                # Main Flask application
├── database.py           # SQLite database connection
├── utils.py              # Short code generator
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── .gitignore
└── urls.db               # SQLite database (generated automatically)
```

---

## ⚙ Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hack-o-week-oddSem-2026-27.git
```

### Navigate to the Project

```bash
cd url-shortener-api
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The server starts at:

```
http://127.0.0.1:5000
```

---

# 📖 API Documentation

---

## 1️⃣ Home

### GET /

Returns a welcome message.

### Response

```json
{
    "success": true,
    "message": "Welcome to URL Shortener API"
}
```

---

## 2️⃣ Shorten URL

### POST /shorten

Creates a shortened URL.

### Request Body

```json
{
    "url":"https://google.com"
}
```

### Success Response

```json
{
    "success": true,
    "message": "Short URL created successfully",
    "original_url":"https://google.com",
    "short_code":"Ab12Cd",
    "short_url":"http://127.0.0.1:5000/Ab12Cd"
}
```

---

## Invalid URL Response

```json
{
    "success": false,
    "message":"Invalid URL"
}
```

---

## Duplicate URL Response

```json
{
    "success": true,
    "message":"URL already shortened",
    "short_code":"Ab12Cd",
    "short_url":"http://127.0.0.1:5000/Ab12Cd"
}
```

---

## 3️⃣ Redirect

### GET /<short_code>

Redirects the user to the original URL.

Example

```
http://127.0.0.1:5000/Ab12Cd
```

Automatically redirects to

```
https://google.com
```

---

## 4️⃣ Analytics

### GET /analytics/<short_code>

Returns analytics for a shortened URL.

Example Response

```json
{
    "original_url":"https://google.com",
    "short_code":"Ab12Cd",
    "clicks":5
}
```

---

## 5️⃣ Delete URL

### DELETE /delete/<short_code>

Deletes a shortened URL.

Example Response

```json
{
    "success": true,
    "message":"Short URL deleted successfully"
}
```

---

# 🗄 Database Schema

The application uses SQLite.

Table: **urls**

| Column | Type |
|---------|------|
| id | INTEGER |
| short_code | TEXT |
| original_url | TEXT |
| clicks | INTEGER |

---

# 🧪 Testing

The API was tested using:

- Thunder Client
- REST API requests
- Browser redirects

---

# 🚀 Future Improvements

- Custom short URLs
- QR Code generation
- User authentication
- URL expiration
- Dashboard UI
- Docker support
- Deployment on Render/Railway
- Swagger/OpenAPI documentation
- Rate limiting
- User accounts and history

---

# 🎯 Learning Outcomes

This project helped in understanding:

- RESTful API development
- Flask routing
- HTTP methods (GET, POST, DELETE)
- SQLite database operations
- CRUD implementation
- Input validation
- Redirect handling
- JSON API responses
- Git & GitHub workflow

---

# 👩‍💻 Author

**Shravani Sadawarte**

B.Tech Computer Science (Data Science)

Hack-O-Week Backend Project

---

## ⭐ If you found this project useful, consider giving it a star!
