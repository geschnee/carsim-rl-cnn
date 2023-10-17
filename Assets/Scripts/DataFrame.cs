using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;
using System.Data;
using System.IO;
using System.Text;

public class DataFrameManager
{
    private DataTable dt;
    private string filePath;
    private int episodeNr;
    public DataFrameManager(string filePath, Boolean isEvaluation)
    {
        this.filePath = filePath;
        this.episodeNr = 0;
        // Initialize the DataTable with column names but no rows
        dt = new DataTable();
        if (!isEvaluation)
        {
            dt.Columns.Add("episodeNr", typeof(int));
            dt.Columns.Add("reward", typeof(double));
            dt.Columns.Add("endEvent", typeof(string));
            dt.Columns.Add("velocity", typeof(double));
            dt.Columns.Add("steps", typeof(int));
            dt.Columns.Add("time", typeof(double));

            if (File.Exists(filePath))
            {
                using (StreamReader sr = new StreamReader(filePath))
                {
                    // Skip the header line
                    sr.ReadLine();

                    while (!sr.EndOfStream)
                    {
                        string[] rows = sr.ReadLine().Split(',');

                        DataRow dr = dt.NewRow();
                        dr["episodeNr"] = Int32.Parse(rows[0]);
                        dr["reward"] = Double.Parse(rows[1]);
                        dr["endEvent"] = rows[2];
                        dr["velocity"] = Double.Parse(rows[3]);
                        dr["steps"] = Int32.Parse(rows[4]);
                        dr["time"] = Double.Parse(rows[5]);
                        this.episodeNr = Int32.Parse(rows[0]);
                        dt.Rows.Add(dr);
                    }
                }
            }
        }

        else
        {

                dt.Columns.Add("map", typeof(string));
                dt.Columns.Add("run", typeof(int));
                dt.Columns.Add("endEvent", typeof(string));
            dt.Columns.Add("passedGoals", typeof(int));
                dt.Columns.Add("velocity", typeof(double));
                dt.Columns.Add("steps", typeof(int));
                dt.Columns.Add("time", typeof(double));
        }

}

    public void AppendRowTraining(int episodeNr, double reward, string endEvent, double velocity, int steps, double time)
    {
        // Add a new row to the DataTable
        this.episodeNr = this.episodeNr + 1;
        this.dt.Rows.Add(this.episodeNr, reward, endEvent, velocity, steps, time);
    }
    public void AppendRowEvaluation(string map, int run, string endEvent, int passedGoals, double velocity, int steps, double time)
    {
        this.dt.Rows.Add(map, run, endEvent, passedGoals, velocity, steps, time);
    }

    public DataTable GetDataTable()
    {
        return dt;
    }

    public int GetEpisodeNr()
    {
        return this.episodeNr;
    }
    public void SaveToCsv()
    {
      
        using (StreamWriter sw = new StreamWriter(this.filePath))
        {
            foreach (DataColumn column in dt.Columns)
            {
                sw.Write($"{column.ColumnName},");
            }

            sw.Write(sw.NewLine);

            foreach (DataRow row in dt.Rows)
            {
                object[] array = row.ItemArray;

                for (int i = 0; i < array.Length - 1; i++)
                {
                    sw.Write($"{array[i].ToString()},");
                }

                sw.Write(array[array.Length - 1].ToString());
                sw.Write(sw.NewLine);
            }

            sw.Close();
        }
    }
}
