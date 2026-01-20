# Frontend Setup Instructions

## Prerequisites
- Node.js and npm installed

## Installation

Dependencies are already installed. If you need to reinstall:
```bash
npm install
```

## Running the Application

### 1. Start JSON Server (Database)
Open a terminal and run:
```bash
npm run json-server
```
This will start the JSON server on `http://localhost:3001` with the database file `db.json`.

### 2. Start the Development Server
Open another terminal and run:
```bash
npm run dev
```
This will start the Vite development server (usually on `http://localhost:5173`).

## Features

- **Home Page**: Displays information about RAG (Retrieval-Augmented Generation) with login/register buttons
- **Authentication**: Login and Register pages with JWT token management
- **Local Database**: JSON Server stores user information locally
- **Routing**: React Router handles navigation between pages

## Database Schema

The `db.json` file stores user data in the following format:
```json
{
  "users": [
    {
      "id": "1",
      "email": "user@example.com",
      "password": "password123",
      "name": "User Name"
    }
  ]
}
```

**Note**: In production, passwords should be hashed. This is a demo implementation.

## Authentication Flow

1. Users can register with email, password, and name
2. Users can login with email and password
3. JWT tokens are stored in localStorage
4. Tokens are validated on each request (client-side validation)

## Next Steps

After completing authentication, you can:
- Add protected routes
- Create a dashboard for authenticated users
- Integrate with the backend RAG system
