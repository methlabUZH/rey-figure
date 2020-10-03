$(document).ready(function(){

    list_leaderboard = "";
    list_images = "";

    window.setInterval(reloadLeaderboard, 5000);
    window.setInterval(reloadImages, 5000);
    reloadLeaderboard();
    reloadImages();

    initIsotope();

});




function reloadLeaderboard(){
    $.ajax({
        url: "ajax/get_leaderboard.php",
        method: "POST",
        data: {list:list_leaderboard}
    }).done(function( response ) {
        var response_parsed = JSON.parse(response);
        list_leaderboard = response_parsed.list;
        $(response_parsed.html).find("tr").each(function(){
            var $row = $(this);
            var score = $(this).data("score");
            var already_there = $(".leaderboard table tr");
            if(already_there.length > 0){
                var lower_exists = false;
                $(".leaderboard table tr").each(function(){
                    if($(this).data('score') < score){
                        console.log("row");
                        console.log($row);
                        console.log("there already");
                        console.log($(this));
                        $row.insertBefore($(this));
                        lower_exists = true;
                        return false;
                    }
                });
                if(!lower_exists){
                    $(".leaderboard table").append($row);
                }
            }
            else{
                $(".leaderboard table").append($row);
            }
        });
        redoPositions();
    });
}

function reloadImages(){
    $.ajax({
        url: "ajax/get_images.php",
        method: "POST",
        data: {list:list_images}
    }).done(function( response ) {
        var response_parsed = JSON.parse(response);
        list_images = response_parsed.list
        var $response = $(response_parsed.html);
        $response.imagesLoaded(function() {
            $( ".images" ).isotope('insert', $response);
        });
    });
}

function redoPositions(){
    var i = 0;
    $("tr").each(function(){
        alert("ok");
    });
}

function initIsotope(){
    $(".images").isotope({
        itemSelector: '.tile',
        layoutMode: 'masonry',
        getSortData: {
            order: "[data-order]"
        },
        sortBy: 'order',
        sortAscending : false
    });
}