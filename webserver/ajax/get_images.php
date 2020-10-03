<?php
/**
 * Created by PhpStorm.
 * User: jheitz
 * Date: 05.12.18
 * Time: 17:14
 */


error_reporting(E_ALL);
ini_set('display_errors', 1);

$images_dir = "../demoimages";

$list = array();
if(isset($_POST['list'])){
    $list = explode("|", $_POST['list']);
}


//$list = ['lskdfj---1.jpg'];
//$entry = 'lskdfj---1.jpg';
//echo (in_array($entry,$list) === false) ? "false" : "true";
//exit();

$log = "";

$filenames = array();
if ($handle = opendir($images_dir)) {
    while (false !== ($entry = readdir($handle))) {
        if ($entry != "." && $entry != "..") {
            $log .= "looking at $entry ";
            if(in_array($entry,$list) === false and substr($entry,0,1) != "."){
                $filenames[filemtime($images_dir."/".$entry) + rand(0,1000)] = $entry;
                $log .= "adding $entry ";
            }
        }
    }
    closedir($handle);
}

arsort($filenames);


ob_start();


?>

<?php foreach($filenames as $time=>$name): ?>
    <div class="tile" data-order="<?= $time; ?>">
        <a href="demoimages/<?= $name; ?>" data-gallery><img src="demoimages/<?= $name; ?>"></a>
    </div>
<?php endforeach; ?>

<?php
/*
?><div class="tile">
    <div class="images-list"><?= implode("|", array_merge($filenames,$list)); ?></div>
</div>
*/
?>

<?php
$html = ob_get_clean();

$new_list = implode("|", array_merge($filenames,$list));

$object = new stdClass();
$object->html = $html;
$object->list = $new_list;
$object->filenames = $filenames;
$object->log = $log;

echo json_encode($object);

?>
