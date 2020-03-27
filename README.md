# Urban-Sound-Classification
## Demo video 
https://drive.google.com/drive/folders/1YD6HC6ZS4aH8pj6sVm5QcA5Ozi817GpX?usp=sharing
## Introduction and Motivation
Over the past five years, developments in artificial intelligence have moved into the medium of sound, whether it be in generating new forms of music(with varying degrees of success), or identifying specific instruments from a video. Some of these projects, like IBM’s Watson Beat, have already been released for commercial use — indeed, it’s creator claims that the network behind Watson Beat has learned the emotional response associated with specific musical elements, something that is strongly subjective and previously the exclusive domain of human composers.

Automatic environmental sound classification is a growing area of research with numerous real-world applications. Whilst there is a large body of research in related audio fields such as speech and music, work on the classification of environmental sounds is comparatively scarce.
Likewise, observing the recent advancements in the field of image classification where convolutional neural networks are used to to classify images with high accuracy and at scale, it begs the question of the applicability of these techniques in other domains, such as sound classification.
 
Although Long-Short Term Memory neural networks (LSTMs) are usually associated with audio-based deep learning projects, elements of sound identification can also be tackled as a traditional image multi-class classification task using convolutional neural networks.

There is a plethora of real world applications for this research, such as:
• Content-based multimedia indexing and retrieval
 • Assisting deaf individuals in their daily activities
 • Smart home use cases such as 360-degree safety and security capabilities
 • Industrial uses such as predictive maintenance
However, while there is a large body of research in related areas such as speech, music and bioacoustics, work on the analysis of urban acoustic environments is relatively scarce.Furthermore, when existent, it mostly focuses on the classification of auditory scene type, e.g. street, park, as opposed to the identification of sound sources in those scenes, e.g.car horn, engine idling, bird tweet.
There are primarily two major challenges with urban sound research namely
Lack of labeled audio data. Previous work has focused on audio from carefully produced movies or television tracks from specific environments such as elevators or office spaces and on commercial or proprietary datasets . The large effort involved in manually annotating real-world data means datasets based on field recordings tend to be relatively small (e.g. the event detection dataset of the IEEE AASP Challenge consists of 24 recordings per each of 17 classes).
Lack of common vocabulary when working on urban sounds.This means the classification of sounds into semantic groups may vary from study to study, making it hard to compare results. 
 
So the objective of this project is to address the above two mentioned challenges.
## PROJECT DESCRIPTION 
 
To apply Deep Learning techniques to the classification problem of environmental sounds, specifically focusing on the identification of particular urban sounds.
When given an audio sample in a computer readable format (such as a .wav file) of a few seconds duration, we want to be able to determine if it contains one of the target urban sounds with a corresponding Classification Accuracy score. We want our model to return various predictions along with the corresponding probability.

## FULL REPORT 
https://docs.google.com/document/d/1jjsGQrt7CF3Hgz0ypewlMFymtC9Ljy6GKR0V0jliQlI/edit?usp=sharing
