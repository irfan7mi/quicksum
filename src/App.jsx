import React, { useState } from "react";
import './App.css';
import Home from './components/Home';
import Login from "./components/Login";
import Signup from "./components/Signup";
import { ToastContainer, toast } from 'react-toastify';
import { StoreProvider } from "./StoreContext";

function App() {
  const [authType, setAuthType] = useState("login");  
  const [isAuthenticated, setIsAuthenticated] = useState(false); 

  return (
    <StoreProvider>
    <div className="App">
      <ToastContainer
      position="top-right"
      autoClose={1000}
      hideProgressBar={false}
      newestOnTop={false}
      closeOnClick={false}
      rtl={false}
      pauseOnFocusLoss
      draggable
      pauseOnHover
      theme="light"
      />
      {!isAuthenticated ? (
        authType === "login" ? (
          <Login setAuthType={setAuthType} setIsAuthenticated={setIsAuthenticated} />
        ) : (
          <Signup setAuthType={setAuthType} setIsAuthenticated={setIsAuthenticated} />
        )
      ) : (
        <Home setIsAuthenticated={setIsAuthenticated}/>
      )}
    </div>
    </StoreProvider>
  );
}

export default App;