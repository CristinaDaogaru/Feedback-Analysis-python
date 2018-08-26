using System;
using System.Linq;
using System.Threading.Tasks;
using System.Web;
using System.Web.Mvc;
using FeedbackPlatform.Models;
using System.Data;
using FeedbackPlatform.Database;
using System.Collections.Generic;

namespace FeedbackPlatform.Controllers
{
    public class ManageController : Controller
    {
        private DataTable dt;

        public ManageController()
        {
        }

        #region Added by Me

        public ActionResult SurveyAddQuestion(int surveyId)
        {
            DatabaseContext dbContext = new DatabaseContext();

            return View();
        }

        [HttpPost]
        [AllowAnonymous]
        public ActionResult CreateNewSurvey(Survey model)
        {
            DatabaseContext dbContext = new DatabaseContext();


            var survey = new Survey
            {
                Name = model.Name,
                CategoryId = model.CategoryId,
                ClientId = model.ClientId
            };

            dbContext.Surveys.Add(survey);
            dbContext.SaveChanges();
            model.Id = survey.Id;
            return View(model);
        }

        [HttpGet]
        [AllowAnonymous]
        public ActionResult EditSurvey(int surveyId, int clientId)
        {
            DatabaseContext dbContext = new DatabaseContext();
            var data = dbContext.Surveys.Where(s => s.Id == surveyId).Where(s => s.ClientId == clientId).ToList();

                Survey model = new Survey
                {
                    Name = data.First().Name,
                    ClientId = data.First().ClientId,
                    Id = data.First().Id,
                    CategoryId = data.First().CategoryId
                };
            

            return View(model);
        }
        #endregion


        #region PrivateMethods
        private void SaveSurvey(int id, string name, int category)
        {
            dt = new DataTable();

        }
        #endregion
    }
}