
var express = require('express');
var app = express();
app.use('/',express.static('public'));

app.set('view engine','hbs' );
app.set('views','views');
app.use('/def',express.static('views')); 
app.use('/model',express.static('views'));

app.listen(3000, function() {
    console.log('server running on port 3000');
} )


app.get('/abc', callName);

function callName(req, res) {

    
    var spawn = require("child_process").spawn;

  
    var process = spawn('python',["./script_audio.py"] );

   
    process.stdout.on('data', function(data) {
        res.send(data.toString());
    } )
}

app.get('/def', callName2);

function callName2(req, res) {
	//console.log(req);
    //console.log(res);
    var spawn = require("child_process").spawn;

  
    var process = spawn('python',["./script_run.py"] );

    // Takes stdout data from script which executed
    // with arguments and send this data to res object
    process.stdout.on('data', function(data) {
		data = data.toString().split('%');
		data.pop();
        res.render('def',{arr:data});
    } )
}

app.get('/model', callName3);

function callName3(req, res) {

    
    res.render('model');
}


// save code as start.js
