import axios from 'axios';
import { jwtDecode } from 'jwt-decode';

const API_URL = 'http://localhost:3001';

export interface User {
  id: string;
  email: string;
  name: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

// Simple JWT token generation (for demo purposes)
// In production, this should be done on the backend
function generateToken(user: User): string {
  const payload = {
    userId: user.id,
    email: user.email,
    name: user.name,
    exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60) // 24 hours
  };
  
  // Simple base64 encoding (not secure, but works for demo)
  // In production, use proper JWT library on backend
  const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const encodedPayload = btoa(JSON.stringify(payload));
  return `${header}.${encodedPayload}.signature`;
}

export const authService = {
  async register(data: RegisterData): Promise<AuthResponse> {
    try {
      // Check if user already exists
      const existingUsers = await axios.get(`${API_URL}/users?email=${data.email}`);
      if (existingUsers.data.length > 0) {
        throw new Error('User with this email already exists');
      }

      // Create new user
      const response = await axios.post(`${API_URL}/users`, {
        email: data.email,
        password: data.password, // In production, hash this!
        name: data.name
      });

      const user: User = {
        id: response.data.id,
        email: response.data.email,
        name: response.data.name
      };

      const token = generateToken(user);
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(user));

      return { token, user };
    } catch (error: any) {
      throw new Error(error.response?.data?.message || error.message || 'Registration failed');
    }
  },

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response = await axios.get(`${API_URL}/users?email=${credentials.email}`);
      
      if (response.data.length === 0) {
        throw new Error('Invalid email or password');
      }

      const userData = response.data[0];
      
      // In production, compare hashed passwords!
      if (userData.password !== credentials.password) {
        throw new Error('Invalid email or password');
      }

      const user: User = {
        id: userData.id,
        email: userData.email,
        name: userData.name
      };

      const token = generateToken(user);
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(user));

      return { token, user };
    } catch (error: any) {
      throw new Error(error.message || 'Login failed');
    }
  },

  logout(): void {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  getCurrentUser(): User | null {
    try {
      const userStr = localStorage.getItem('user');
      if (!userStr) return null;
      return JSON.parse(userStr);
    } catch (error) {
      console.warn('Error parsing user from localStorage:', error);
      localStorage.removeItem('user');
      return null;
    }
  },

  getToken(): string | null {
    return localStorage.getItem('token');
  },

  isAuthenticated(): boolean {
    const token = this.getToken();
    if (!token) return false;

    try {
      const decoded: any = jwtDecode(token);
      // Check if token is expired
      if (decoded.exp && decoded.exp < Date.now() / 1000) {
        this.logout();
        return false;
      }
      return true;
    } catch (error) {
      // If token is invalid, clear it
      console.warn('Invalid token:', error);
      this.logout();
      return false;
    }
  }
};
