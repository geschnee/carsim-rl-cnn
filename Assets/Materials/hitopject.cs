using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class hitopject : MonoBehaviour
{
	[SerializeField] private UnityEngine.UI.Text text;
	[SerializeField] private Transform car;
	[SerializeField] public int count;
    
    private void OnTriggerEnter(Collider other)
    {
	    if (other.tag == "red" && !(isleft(other.transform.position))){
			
		   count++;
		}else if(other.tag == "blue" && isleft(other.transform.position)){
			count++;
		}else{
			Debug.Log("Tag Problem:"+other.tag);
		}
		//text.text = b.ToString();
	    
    }
    
    private bool isleft(Vector3 position){
		Vector3 Dir = position - car.position;
		Dir = Quaternion.Inverse(car.rotation) * Dir;
		return (Dir.x>0);
	}

}
