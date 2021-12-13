const observer = new MutationObserver(function () {
    //checks if the element exists in the DOM
    if (document.getElementById("contenteditable-root")) {
      observer.disconnect()
      //console.log(chrome.i18n.getMessage("comments_disabled"))
      hideComments()
    }
  });


observer.observe(document.documentElement, {
    attributes: true,
    childList: true,
    characterData: false,
    subtree: true
  });

function getPrediction(user_input_text) {
    const url = "http://localhost:3000/predict/"
    const response = fetch(url.concat(user_input_text))
    console.log(response);
    document.getElementById
}
