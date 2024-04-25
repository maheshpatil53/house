const userInput = document.getElementById("area").value;
setCookie("area", userInput, 1);
return false;

const userInput1 = getCookie("userInput");
if (userInput !== "") {
  document.getElementById("printhere").innerHTML = userInput1;
}
