using FeedbackPlatform.Database;
using FeedbackPlatform.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace FeedbackPlatform.Controllers
{
    public class HomeController : Controller
    {
        public ActionResult Index()
        {
            return View();
        }

        public ActionResult About()
        {
            ViewBag.Message = "Your application description page.";

            return View();
        }

        public ActionResult Contact()
        {
            ViewBag.Message = "Your contact page.";

            return View();
        }

        public ActionResult Home(int clientId)
        {
            DatabaseContext dbContext = new DatabaseContext();
            SurveyModelControl surveyModelControl = new SurveyModelControl();
            surveyModelControl.Survey = new Survey();

            var data = dbContext.Surveys.Where(s => s.ClientId == clientId).ToList();
            
            foreach(var d in data)
            {
                Survey survey = new Survey
                {
                    Name = d.Name,
                    Id = d.Id,
                    ClientId = d.ClientId,
                    CategoryId = d.CategoryId,
                    QuestionId = d.QuestionId
                };
                surveyModelControl.Surveys.Add(survey);
            }
            surveyModelControl.Survey.ClientId = clientId;
            var questionNumber = dbContext.Questions.Where(q => q.Id == surveyModelControl.Survey.QuestionId);
            surveyModelControl.QuestionNumber = questionNumber.Count();
            return View(surveyModelControl);
        }
    }
}