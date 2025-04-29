import { createContext, useContext, useState } from "react";

const StoreContext = createContext();

export const StoreProvider = ({ children }) => {
  const [email, setEmail] = useState("");

  return (
    <StoreContext.Provider value={{ email, setEmail}}>
      {children}
    </StoreContext.Provider>
  );
};

export const useStore = () => useContext(StoreContext);
