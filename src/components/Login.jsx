import React, { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import { useStore } from "../StoreContext";

const Login = ({ setAuthType, setIsAuthenticated }) => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const { email, setEmail } = useStore();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };
  console.log(email)
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/login", formData, {
        headers: { "Content-Type": "application/json" },
        withCredentials: true, // ✅ Allow credentials (important for CORS)
      });
      toast.success(response.data.message)
      setIsAuthenticated(true); // ✅ Set authenticated to true on success
    } catch (error) {
      toast.error("Login Failed")
    }
  };

  return (
    <div className="auth-container">
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <input type="email" name="email" placeholder="Email" onChange={(e) => {
    handleChange(e);  // Call handleChange function properly
    setEmail(e.target.value);  // Update state
     }} required />
        <input type="password" name="password" placeholder="Password" onChange={handleChange} required />
        <button type="submit">Sign In</button>
        <p className="switch-auth" onClick={() => setAuthType("signup")}>
          Don't have an account? Sign up
        </p>
      </form>
    </div>
  );
};

export default Login;