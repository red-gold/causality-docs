console.log('this is run examples');
var s = document.createElement("script");
s.type = "text/javascript";
s.src = "https://embed.runkit.com";
var babel = document.createElement("script");
babel.type = "text/javascript";
babel.src = "https://unpkg.com/@babel/standalone/babel.min.js";
$("head").append(babel);
$.when( $.ready ).then(function() {
    $('*a:contains("Run code")').each(function() {
        $(this).addClass('pure-button');
        $(this).attr('disabled');
        $(this).on('click', function(event){
            event.preventDefault();
            alert('this function will be updated soon');
        });
        const Link = $(this).attr("href");
        const Name = Link.split(/[\/]+/).pop().replace(/\./g,'_');
        console.log({Name, Link});
        // $.get(Link, (data, status)=>{
        //     $(`#${Name}`).html =  data;    
        //     $( `<div id="${Name}">${data}</div>` ).insertAfter( $(this) );
        // });
    });
});
