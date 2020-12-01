<?php
/**
 * Created by PhpStorm.
 * User: jheitz
 * Date: 05.12.18
 * Time: 16:41
 */
error_reporting(E_ALL);
ini_set('display_errors', 1);

$images_dir = "../demoimages";

$list = array();
if(isset($_POST['list'])){
    $list = explode("|", $_POST['list']);
}

$filenames = array();
if ($handle = opendir($images_dir)) {
    while (false !== ($entry = readdir($handle))) {
        if ($entry != "." && $entry != "..") {
            if(in_array($entry,$list) === false)
                array_push($filenames,$entry);
        }
    }
    closedir($handle);
}



$images = array();
foreach($filenames as $img){
    $parts = explode("---", $img);
    if(count($parts) < 3) continue;
    $title = implode("---", array_slice($parts, 0, -2));
    $score_rounded = $parts[count($parts) - 2];
    $score = implode(".", array_slice(explode(".", end($parts)),0,-1));
    if($title == "" or $score=="") continue;
    $images[$title] = array("score" => $score, "score_rounded" => $score_rounded, "filename" => $img);
}

arsort($images);

ob_start();


?>

<table >
    <?php foreach($images as $name => $arr): ?>
    <tr data-score="<?= $arr['score']; ?>">
        <td></td>
        <td><a href="demoimages/<?= $arr['filename']; ?>" data-gallery><?= $name; ?></a></td>
        <td><?= $arr['score_rounded']; ?></td>
    </tr>
    <?php endforeach; ?>
</table>

<?php
$html = ob_get_clean();

$new_list = implode("|", array_merge($filenames,$list));

$object = new stdClass();
$object->html = $html;
$object->list = $new_list;

echo json_encode($object);

?>

