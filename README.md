# Multimodal_ROS_Pipeline
This multimodal pipeline was developed to facilitate Human-Robot Interaction in outdoor collaborative environments, particularly in the table-grape vineyard of the [CANOPIES](https://canopies.inf.uniroma3.it/) project. The system encompasses speech and gesture modalities for Human-To-Robot Interaction, consisting of 9 modules.  


## Technical Aspects
* Ubuntu 20.04 LTS
* ROS Noetic
* Python 3.8.10 


## Getting Started

### Prerequisites
* Create a ROS workspace
* Access the workspace and download the required data to run the system, available at this [link](https://drive.google.com/drive/folders/1D6eX82cT9xsVohmB5VHu35TXJCwXFaH7?usp=sharing) 

### Installation
* Clone this repository in your ROS workspace:

```
git clone https://github.com/Sapienza-HRI/Multimodal_ROS_Pipeline
```

Remember to compile and source the code before running it!


## Run
You can choose to run the complete multimodal system, only the speech pipeline, or the gesture architecture, depending on which acquisition and recognition modules you launch.



## Citation
If you are using this code in your project or research, please cite these articles:

```
@InProceedings{10.1007/978-981-99-8718-4_9,
author="Kaszuba, Sara
and Caposiena, Julien
and Sabbella, Sandeep Reddy
and Leotta, Francesco
and Nardi, Daniele",
editor="Ali, Abdulaziz Al
and Cabibihan, John-John
and Meskin, Nader
and Rossi, Silvia
and Jiang, Wanyue
and He, Hongsheng
and Ge, Shuzhi Sam",
title="Empowering Collaboration: A Pipeline for Human-Robot Spoken Interaction in Collaborative Scenarios",
booktitle="Social Robotics",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="95--107",
isbn="978-981-99-8718-4"
}
```

and 

```
@InProceedings{10.1007/978-981-99-8718-4_14,
author="Sabbella, Sandeep Reddy
and Kaszuba, Sara
and Leotta, Francesco
and Nardi, Daniele",
editor="Ali, Abdulaziz Al
and Cabibihan, John-John
and Meskin, Nader
and Rossi, Silvia
and Jiang, Wanyue
and He, Hongsheng
and Ge, Shuzhi Sam",
title="Gesture Recognition for Human-Robot Interaction Through Virtual Characters",
booktitle="Social Robotics",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="160--170",
isbn="978-981-99-8718-4"
}
```
